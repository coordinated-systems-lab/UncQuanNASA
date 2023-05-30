"""Adapted from https://github.com/microsoft/LightGBM/issues/5727#issue-1589436779"""

from typing import Iterable, List, Union, Tuple
from functools import partial

import numpy as np

import lightgbm as lgb

import pandas as pd
import optuna
import pynumdiff
import pynumdiff.optimize

from sklearn.model_selection import train_test_split


def _calc_tvgamma(freq: float, dt: float):
    log_gamma = -1.6 * np.log(freq) - 0.71 * np.log(dt) - 5.1
    return np.exp(log_gamma)


def smooth_series(
    x: np.ndarray, dt: float, freq: float
) -> Tuple[np.ndarray, np.ndarray]:
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
    return x_hat[1:-1] - x_hat[:-2], dxdt_hat_hat[2:] - dxdt_hat_hat[1:-1]


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.Dataset, alphas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = dtrain.get_label()
    grad = (y_pred > y_train).astype(int) - alphas
    hess = np.ones(y_train.shape)
    return grad, hess


def pinball_metric(
    preds: np.ndarray, eval_data: lgb.Dataset
) -> Tuple[str, float, bool]:
    # Get alpha values
    alphas = eval_data.get_data()[:, -1]

    # Calculate loss
    u = eval_data.get_label() - preds
    result = (u * (alphas - (u < 0).astype(int))).mean()

    return ("pinball", result, False)


def objective(trial, data: lgb.Dataset):
    # Get datasets out
    X = data.get_data()
    y = data.get_label()

    # Split train test randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Redefine lgb datasets
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, free_raw_data=False).construct()

    param = {
        "verbosity": -1,
        "boosting_type": "gbdt",
        "metric": "pinball",
        "num_iterations": trial.suggest_int("num_iterations", 1000, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-3),
        # "early_stopping_round": trial.suggest_int("use_early_stopping", 0, 0)
        # * trial.suggest_int("early_stopping_round", 1, 100),
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
        "mc": [0 for _ in range(X.shape[1] - 1)] + [1],
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "pinball")
    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dtest],
        callbacks=[pruning_callback],
        fobj=partial(check_loss_grad_hess, alphas=X_train[:, -1]),
        feval=pinball_metric,
    )

    # Save booster
    trial.set_user_attr(key="best_booster", value=gbm)

    # Return performance
    preds = gbm.predict(X_test)
    _, perf, _ = pinball_metric(preds, dtest)
    return perf


def save_best_model(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


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
        random_state: Union[int, None] = None,
    ):
        self.n_feat = x.shape[1]

        # Add features
        x = _feature_transform(x)

        # Smooth target and 2nd derivatives
        self.y_all = np.zeros((x.shape[0] - 2, self.n_feat))
        x_ = np.zeros((x.shape[0] - 2, self.n_feat))
        for i in range(self.n_feat):
            x_[:, i], self.y_all[:, i] = smooth_series(x[:, i], dt=dt, freq=freq)

        x = x[1:-1, :]
        x = np.hstack((x, x_))

        # Get bootstrapped rows
        self.rng = np.random.default_rng(random_state)
        n = round(m_factor * x.shape[0])
        idx = self.rng.choice(x.shape[0], size=n, replace=True)

        # reindex, add alphas
        self.x_train = x[idx, :]
        self.y_train = self.y_all[idx, :]
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

    def train(self) -> List[lgb.Booster]:
        self.models = []
        for i in range(len(self.datasets)):
            print(f"Running hyperparameter tuning for model {i + 1}...")

            # Old way
            curr_obj = partial(objective, data=self.datasets[i])
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            )
            study.optimize(curr_obj, n_trials=1, callbacks=[save_best_model])

            # Results
            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            self.models.append(study.user_attrs["best_booster"])

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
                x_[1:-1, i], _ = smooth_series(x[:, i], self.dt, self.freq)
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
        return preds
