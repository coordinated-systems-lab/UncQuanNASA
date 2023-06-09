"""Adapted from https://github.com/microsoft/LightGBM/issues/5727#issue-1589436779"""

from typing import Any, Iterable, List, Union, Tuple
from functools import partial

import numpy as np

import lightgbm as lgb

import pandas as pd
import pynumdiff
import pynumdiff.optimize


def _calc_tvgamma(freq: float, dt: float):
    log_gamma = -1.6 * np.log(freq) - 0.71 * np.log(dt) - 5.1
    return np.exp(log_gamma)


def smooth_series(
    x: np.ndarray, dt: float, freq: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Calculate optimal parameter for optimization
    tvgamma = _calc_tvgamma(freq, dt)

    params, _ = pynumdiff.optimize.smooth_finite_difference.butterdiff(
        x, dt, params=None, options={"iterate": True}, tvgamma=tvgamma, dxdt_truth=None
    )
    # print('Optimal parameters: ', params)
    x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(
        x, dt, params, options={"iterate": True}
    )
    dxdt_hat_hat, ddxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(
        dxdt_hat, dt, params, options={"iterate": True}
    )

    # return ddxdt_hat
    return x_hat[1:-1], x_hat[1:-1] - x_hat[:-2], dxdt_hat_hat[2:] - dxdt_hat_hat[1:-1]


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.Dataset, alphas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = dtrain.get_label()
    grad = (y_pred > y_train).astype(int) - alphas
    hess = np.ones(y_train.shape)
    return grad, hess


def _feature_transform(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    # Convert type
    if isinstance(x, pd.DataFrame | pd.Series):
        x = x.to_numpy()
    return x


class QuantileRegressionSimulator:
    """Second Order Quantile Regression Simulator

    This class does the following:

    1. Takes in n state vectors
    2. Estimates second derivative between every point and the next one; these are the
        target.
    3. Estimates first derivative between every point and the previous one; these gets
        added as features.
    4. Add a quantile feature, which allows model to estimate entire distribution
    5. Fit n models, using all states and first derivatives as features.
    6. Given a starting state and first derivative, estimate trajectories by
        iteratively making predictions from model and sampling from output
    """

    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        m_factor: float,
        # alphas: list[float],
        n_diff: int,
        dt: float,
        freq: float,
        model_params: dict[str, Any],
        random_state: Union[int, None] = None,
        seq_id: Union[pd.Series, np.ndarray, None] = None,
    ):
        self.n_feat = x.shape[1]

        # Add features
        x = _feature_transform(x)

        # Smooth target and 2nd derivatives
        if seq_id is None:
            seq_id = np.zeros(x.shape[0])

        x_list, d1_list, d2_list, resid_list = [], [], [], []
        for id in np.unique(seq_id):
            mask = seq_id == id
            n = mask.sum()

            x_hat = np.zeros((n - 2, self.n_feat))
            d1 = np.zeros((n - 2, self.n_feat))
            d2 = np.zeros((n - 2, self.n_feat))
            for i in range(self.n_feat):
                x_hat[:, i], d1[:, i], d2[:, i] = smooth_series(
                    x[mask, i], dt=dt, freq=freq
                )

            resid_list.append(x[mask, :][1:-1, :] - x_hat)
            x_list.append(x[mask, :][1:-1, :])
            d1_list.append(d1)
            d2_list.append(d2)

        x = np.hstack((np.vstack(x_list), np.vstack(d1_list)))
        self.resids = np.vstack(resid_list)
        self.d2 = np.vstack(d2_list)

        # Get bootstrapped rows
        self.rng = np.random.default_rng(random_state)
        n = round(m_factor * x.shape[0])
        idx = self.rng.choice(x.shape[0], size=n, replace=True)

        # reindex, add alphas
        self.x_train = x[idx, :]
        self.y_train = self.d2[idx, :]
        alphas = self.rng.uniform(size=(n, 1))
        self.x_train = np.hstack((self.x_train, alphas))

        # Lightgbm preps
        self.datasets = []
        for i in range(self.n_feat):
            self.datasets.append(
                lgb.Dataset(
                    data=self.x_train,
                    label=self.y_train[:, i],
                    free_raw_data=False,
                ).construct()
            )

        # Save params
        self.rng = np.random.default_rng(random_state)
        self.alphas = alphas
        self.n_diff = n_diff
        self.dt = dt
        self.freq = freq
        self.model_params = model_params

    def train(self) -> List[lgb.Booster]:
        self.models = []
        for i in range(len(self.datasets)):
            # Get datasets out
            X = self.datasets[i].get_data()
            y = self.datasets[i].get_label()

            # Train data
            dtrain = lgb.Dataset(X, label=y, free_raw_data=False)

            # Train model
            gbm = lgb.train(
                self.model_params,
                dtrain,
                fobj=partial(check_loss_grad_hess, alphas=X[:, -1]),
            )
            self.models.append(gbm)

        return self.models

    def predict(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        alpha: Iterable[float] | float,
        smooth: bool = True,
    ) -> np.ndarray:
        if isinstance(alpha, float):
            alpha = [alpha for _ in self.models]
        alpha = list(alpha)

        _pred = np.zeros((x.shape[0], self.n_feat))

        # Feature engineer and clean
        x = _feature_transform(x)

        # Add first derivatives
        if smooth:
            x_ = np.zeros((x.shape[0], self.n_feat))
            for i in range(self.n_feat):
                _, x_[1:-1, i], _ = smooth_series(x[:, i], self.dt, self.freq)
            x = np.hstack((x, x_))

        # Make predictions
        for i in range(len(self.models)):
            _x = np.hstack((x, np.repeat(alpha[i], x.shape[0]).reshape(-1, 1)))
            _pred[:, i] = self.models[i].predict(_x)

        return _pred

    def sol_fn(self, X0: np.ndarray) -> np.ndarray:
        if self.sample:
            # runif = self.rng.uniform(size=len(self.models))
            runif = np.repeat(self.rng.uniform(), len(self.models))
            # runif = self.rng.uniform()
        else:
            runif = np.repeat(0.5, len(self.models))
            # runif = 0.5

        if len(X0.shape) == 1:
            X0 = X0.reshape(1, -1)

        # Make predictions
        pred = self.predict(X0, runif, smooth=False)
        return pred[0, :]

    def simulate_path(
        self,
        X0: np.ndarray,
        steps: int,
        sample: bool = True,
    ) -> np.ndarray:
        self.sample = sample
        preds = np.zeros((steps, self.n_feat))
        curr_pos = X0.copy()

        for i in range(steps):
            preds[i, :] = curr_pos[: self.n_feat]
            # hold_for_prev_pos = curr_pos[:self.n_feat].copy()
            if self.n_diff == 0:
                curr_pos = self.sol_fn(curr_pos)
            elif self.n_diff == 1:
                curr_pos += self.sol_fn(curr_pos)  # * self.dt
            elif self.n_diff == 2:
                der2_add = self.sol_fn(curr_pos) * self.dt  # ** 2
                # print(f"{curr_pos = }")
                # print(f"{der2_add = }")
                # print()
                curr_der1 = curr_pos[self.n_feat:] + der2_add
                curr_pos[: self.n_feat] += curr_der1
                # prev_pos = hold_for_prev_pos
                curr_pos[self.n_feat:] = curr_der1.copy()

        # Add back in noise
        for j in range(preds.shape[1]):
            preds[:, j] += self.rng.choice(
                self.resids[:, j], size=preds.shape[0], replace=True
            )

        return preds
