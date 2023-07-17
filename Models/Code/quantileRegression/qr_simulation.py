"""Adapted from https://github.com/microsoft/LightGBM/issues/5727#issue-1589436779"""

from typing import Any, Iterable, List, Union, Tuple
from functools import partial
from tqdm import tqdm

import numpy as np

import lightgbm as lgb

import pandas as pd
import pynumdiff
import pynumdiff.optimize


def _calc_tvgamma(freq: float, dt: float):
    log_gamma = -1.6 * np.log(freq) - 0.71 * np.log(dt) - 5.1
    return np.exp(log_gamma)


def _est_smoothing_params(x: np.ndarray, dt: float, freq: float, smoother: str) -> List:
    # Calculate optimal parameter for optimization
    tvgamma = _calc_tvgamma(freq, dt)

    params, _ = getattr(pynumdiff.optimize.smooth_finite_difference, smoother)(
        x, dt, params=None, options={"iterate": True}, tvgamma=tvgamma, dxdt_truth=None
    )

    return params


def smooth_series(
    x: np.ndarray,
    dt: float,
    params: list,
    smoother: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hat, dxdt_hat = getattr(pynumdiff.smooth_finite_difference, smoother)(
        x, dt, params, options={"iterate": True}
    )
    dxdt_hat_hat, ddxdt_hat = getattr(pynumdiff.smooth_finite_difference, smoother)(
        dxdt_hat, dt, params, options={"iterate": True}
    )

    # return ddxdt_hat
    return x_hat[1:-1], x_hat[1:-1] - x_hat[:-2], dxdt_hat_hat[2:] - dxdt_hat_hat[1:-1]


def sample_smooth_series(
    x: np.ndarray,
    dt: float,
    params: list,
    rng: np.random.Generator,
    n_samples: int,
    perc_sample: float,
    smoother: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = (len(x) - 2) * n_samples
    x_hat, x1, x2 = np.zeros(n), np.zeros(n), np.zeros(n)

    # Average differences
    diff_std = (x[1:] - x[-1]).std() / 3

    # Get smoothed series
    print("Getting smooth sampled series")
    for i in tqdm(range(n_samples)):
        # Get current sample
        curr_ind = (
            rng.choice(
                len(x) - 1, size=round((1 - perc_sample) * len(x)), replace=False
            )
            + 1
        )

        # Copy and remove indices
        curr_x = x.copy()
        curr_x[curr_ind] = np.nan

        # Fill in missing
        curr_x = pd.Series(curr_x).interpolate().to_numpy()

        # Add in some noise
        curr_x[curr_ind] = curr_x[curr_ind] + rng.normal(
            scale=diff_std, size=len(curr_ind)
        )

        # Get smoothed series
        ind = np.arange(i * (len(x) - 2), (i + 1) * (len(x) - 2), dtype=int)
        x_hat[ind], x1[ind], x2[ind] = smooth_series(curr_x, dt, params, smoother)
    return x_hat, x1, x2


def finite_diff(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = (x[1:] - x[:-1]) / dt
    x2 = (x1[1:] - x1[:-1]) / dt

    return x[1:-1], x1[:-1], x2


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.Dataset, alphas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = dtrain.get_label()
    grad = (y_pred > y_train).astype(int) - alphas
    hess = np.ones(y_train.shape)
    return grad, hess


def _check_x(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Tuple[np.ndarray, list]:
    # Convert type to numpy
    feature_names = list(range(x.shape[1]))
    if isinstance(x, pd.DataFrame | pd.Series):
        feature_names = x.columns.tolist()
        x = x.to_numpy()

    return x, feature_names


def _feature_transform(
    x: np.ndarray, feature_names: list
) -> Tuple[np.ndarray, np.ndarray]:
    # Convert theta
    idx = feature_names.index("theta")
    x[:, idx] = np.mod(x[:, idx], 2 * np.pi)

    # x to constant
    keep_feats = ["theta", "theta_d", "x_d", "alpha"]
    return x, np.array([i for i, f in enumerate(feature_names) if f in keep_feats])
    # idx = feature_names.index("x")
    # return x, np.delete(np.arange(x.shape[1]), idx)


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
        alpha_dist: str = "uniform",
        smooth_derv_est: bool = True,
        smoothing_samples: int | None = None,
        smoothing_perc: float = 0.8,
        alpha_dist_params: dict | None = None,
        random_state: Union[int, None] = None,
        seq_id: Union[pd.Series, np.ndarray, None] = None,
        smoother: str = "butterdiff",
    ):
        self.nrow = x.shape[0]
        self.n_feat = x.shape[1]
        self.rng = np.random.default_rng(random_state)

        # Add features
        x, self.feature_names_ = _check_x(x)
        self.feature_names_ += [f"{f}_d" for f in self.feature_names_]

        # Smooth target and 2nd derivatives
        if seq_id is None:
            seq_id = np.zeros(x.shape[0])

        x_list, d1_list, d2_list, resid_list = [], [], [], []
        for id in np.unique(seq_id):
            mask = seq_id == id
            if smoothing_samples is None or not smooth_derv_est:
                n = mask.sum() - 2
            else:
                n = (mask.sum() - 2) * smoothing_samples

            x_hat = np.zeros((n, self.n_feat))
            d1 = np.zeros((n, self.n_feat))
            d2 = np.zeros((n, self.n_feat))
            for i in range(self.n_feat):
                if smooth_derv_est:
                    self.smooth_params = _est_smoothing_params(
                        x[mask, i], dt=dt, freq=freq, smoother=smoother
                    )
                    if smoothing_samples is None:
                        x_hat[:, i], d1[:, i], d2[:, i] = smooth_series(
                            x[mask, i],
                            dt=dt,
                            params=self.smooth_params,
                            smoother=smoother,
                        )
                    else:
                        x_hat[:, i], d1[:, i], d2[:, i] = sample_smooth_series(
                            x[mask, i],
                            dt=dt,
                            params=self.smooth_params,
                            rng=self.rng,
                            n_samples=smoothing_samples,
                            perc_sample=smoothing_perc,
                            smoother=smoother,
                        )
                else:
                    x_hat[:, i], d1[:, i], d2[:, i] = finite_diff(x[mask, i], dt=dt)

            if x_hat.shape[0] == mask.sum():
                resid_list.append(x[mask, :][1:-1, :] - x_hat)
            x_list.append(x_hat)
            # x_list.append(x[mask, :][1:-1, :])
            d1_list.append(d1)
            d2_list.append(d2)

        self.x = np.hstack((np.vstack(x_list), np.vstack(d1_list)))
        x = self.x
        if len(resid_list) > 0:
            self.resids = np.vstack(resid_list)
        else:
            self.resids = np.zeros((self.nrow, self.n_feat))
        self.d2 = np.vstack(d2_list)

        # Get bootstrapped rows
        n = round(m_factor * self.nrow)
        idx = self.rng.choice(x.shape[0], size=n, replace=True)

        # reindex
        self.x_train = x[idx, :]
        self.y_train = self.d2[idx, :]

        # Add alphas
        if alpha_dist_params is None:
            alpha_dist_params = {}
        alphas = getattr(self.rng, alpha_dist)(size=(n, 1), **alpha_dist_params)
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
        self.smoother = smoother
        self.model_params = model_params

    def train(self) -> List[lgb.Booster]:
        self.models = []
        for i in range(len(self.datasets)):
            # Get datasets out
            X = self.datasets[i].get_data()
            y = self.datasets[i].get_label()

            # feature transform
            X, f_idx = _feature_transform(X, self.feature_names_ + ["alpha"])

            # Train data
            dtrain = lgb.Dataset(X[:, f_idx], label=y, free_raw_data=False)

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

        # Clean
        x, _ = _check_x(x)

        # Add first derivatives
        if smooth:
            x_ = np.zeros((x.shape[0], self.n_feat))
            for i in range(self.n_feat):
                curr_smooth_params = _est_smoothing_params(
                    x[:, i], self.dt, self.freq, self.smoother
                )
                _, x_[1:-1, i], _ = smooth_series(
                    x[:, i], self.dt, curr_smooth_params, self.smoother
                )
            x = np.hstack((x, x_))

        # Make predictions
        for i in range(len(self.models)):
            # Add alpha
            _x = np.hstack((x, np.repeat(alpha[i], x.shape[0]).reshape(-1, 1)))

            # Feature transform
            _x, f_idx = _feature_transform(_x, self.feature_names_ + ["alpha"])
            _pred[:, i] = self.models[i].predict(_x[:, f_idx])

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

        # Instantiate array to hold predictions
        preds = np.zeros((steps, self.n_feat))

        # Starting position
        curr_pos = X0.copy()

        for i in range(steps):
            preds[i, :] = curr_pos[: self.n_feat]
            if self.n_diff == 0:
                curr_pos = self.sol_fn(curr_pos)
            elif self.n_diff == 1:
                curr_pos += self.sol_fn(curr_pos)  # * self.dt
            elif self.n_diff == 2:
                # Get current second derivative
                der2_add = self.sol_fn(curr_pos) * self.dt

                # Add to previous first derivative to get new first derivative
                curr_pos[self.n_feat :] = curr_pos[self.n_feat :] + der2_add

                # Add new first derivative to position to get current position
                curr_pos[: self.n_feat] += curr_pos[self.n_feat :]

        # Add back in noise
        for j in range(preds.shape[1]):
            preds[:, j] += self.rng.choice(
                self.resids[:, j], size=preds.shape[0], replace=True
            )

        return preds

    def simulate_paths(
        self,
        X0: np.ndarray,
        n: int,
        steps: int,
        sample: bool = True,
    ) -> np.ndarray:
        self.sample = sample

        # Instantiate array to hold predictions
        preds = np.zeros((n, steps, self.n_feat))

        # Fill in n starting positions
        curr_pos = np.zeros((n, len(X0)))
        curr_pos[:] = X0
        der2 = np.zeros((n, self.n_feat))

        for i in range(steps):
            preds[:, i, :] = curr_pos[:, : self.n_feat].copy()
            if self.n_diff == 2:
                # Add alpha
                x = np.hstack((curr_pos, self.rng.uniform(size=(n, 1))))

                # Feature transform
                x, f_idx = _feature_transform(x, self.feature_names_ + ["alpha"])

                # Make 2nd derivative predictions
                for i in range(self.n_feat):
                    der2[:, i] = self.models[i].predict(x[:, f_idx]) * self.dt

                # Propogate
                curr_pos[:, self.n_feat :] += der2
                curr_pos[:, : self.n_feat] += curr_pos[:, self.n_feat :]
            else:
                raise ValueError(f"{self.n_diff = } currently unsupported here")

        # Add back in noise
        for j in range(self.n_feat):
            preds[:, :, j] += self.rng.choice(
                self.resids[:, j], size=(n, steps), replace=True
            )

        return preds
