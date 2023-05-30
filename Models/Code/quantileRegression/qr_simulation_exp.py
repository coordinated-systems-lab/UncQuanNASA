"""Adapted from https://github.com/microsoft/LightGBM/issues/5727#issue-1589436779"""

from typing import Iterable, List, Union, Tuple
from functools import partial

import numpy as np

import lightgbm as lgb

import pandas as pd
import matplotlib.pyplot as plt
import optuna
import pynumdiff
import pynumdiff.optimize


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
    # dxdt_hat_hat, ddxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(
    #     dxdt_hat, dt, params, options={"iterate": True}
    # )

    return x_hat  # , dxdt_hat_hat, ddxdt_hat


def _grad_rho(u, alpha) -> np.ndarray:
    return -(alpha - (u < 0).astype(float))


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.Dataset, alphas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = dtrain.get_label()
    grad = _grad_rho(y_train - y_pred, alphas)
    hess = np.ones(y_train.shape)
    return grad, hess


def _create_helper_mat(n: int) -> np.ndarray:
    # create a 2D array of zeros with shape (n, n)
    a = np.tile(np.arange(n), (n, 1))
    b = a.copy().T

    res = np.maximum(b - a + 1, 0)
    res[:, :2] = 0
    return res


def pinball_metric(
    preds: np.ndarray, eval_data: lgb.Dataset
) -> Tuple[str, float, bool]:
    # Get alpha values
    alphas = eval_data.get_data()[:, -1]

    # Calculate loss
    u = eval_data.get_label() - preds
    result = (u * (alphas - (u < 0).astype(int))).mean()

    return ("pinball", result, False)


def _calc_trajectory(
    y_pred: np.ndarray, y: np.ndarray, helper_mat: np.ndarray
) -> np.ndarray:
    # n = len(y)

    # Get predicted trajectory
    # diff_1 = y[1] - y[0]
    return (
        y[1]
        # + (np.arange(n) - 1) * diff_1
        + (helper_mat @ y_pred.reshape(-1, 1)).flatten()
    )


def traj_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.Dataset, alphas: np.ndarray, helper_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Get actual trajectory
    y_train = dtrain.get_label()

    # Initialize gradients
    grad = np.zeros(y_train.shape)
    hm = helper_mat.T.copy()
    for alpha in np.unique(alphas):
        mask = alphas == alpha

        # Get predicted trajectory
        t_pred = _calc_trajectory(y_pred[mask] / 100, y_train[mask], helper_mat)
        y_dd = 100 * (y_train[mask][2:] + y_train[mask][:-2] - 2 * y_train[mask][1:-1])

        # Primary gradient - numerical 2nd derivative
        curr_grad1 = np.zeros(mask.sum())
        curr_grad1[2:] = (y_pred[mask][2:] > y_dd).astype(int) - alpha

        # Secondary gradient - trajectory difference
        curr_grad2 = (t_pred > y_train[mask]).astype(int) - alpha
        curr_grad2[:2] = 0
        curr_grad2 = (hm @ curr_grad2.reshape(-1, 1)).flatten()

        # Combine
        mp = 0.05
        curr_grad2 = (
            mp * curr_grad2 * np.abs(curr_grad1).max() / np.abs(curr_grad2).max()
        )
        curr_grad = curr_grad1 + curr_grad2

        # Save
        grad[mask] = curr_grad

        if alpha == 0.5:
            tt = pd.Series(curr_grad[2:])

            # fig, ax = plt.subplots()
            # ax.plot(np.arange(2, len(tt) + 2), tt)
            # ax.set_title("Gradient")
            # plt.show()
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(tt) + 2), t_pred)
            ax.plot(np.arange(len(tt) + 2), y_train[mask])
            ax.set_title("Actual trajectory")
            plt.show()
            fig, ax = plt.subplots()
            # ax.plot(np.arange(len(tt)), y_pred[mask][2:])
            # ax.plot(np.arange(len(tt)), y_dd)
            ax.plot(np.arange(2000), y_pred[mask][2:][:2000])
            ax.plot(np.arange(2000), y_dd[:2000])
            # ax.plot(np.arange(len(tt)), (y_dd - y_pred[mask][2:]).cumsum())
            # ax.plot(np.arange(len(tt)), (y_dd - y_pred[mask][2:]).cumsum().cumsum())
            ax.set_title("actual pred comps")
            plt.show()

    hess = np.ones(y_train.shape)
    return grad, hess


def traj_pinball_metric(
    preds: np.ndarray, eval_data: lgb.Dataset, helper_mat: np.ndarray
) -> Tuple[str, float, bool]:
    # Get actual trajectory
    y_test = eval_data.get_label()

    # Get alpha values
    alphas = eval_data.get_data()[:, -1]

    result = 0
    for alpha in np.unique(alphas):
        mask = alphas == alpha

        # Get predicted trajectory
        t_pred = _calc_trajectory(preds[mask], y_test[mask], helper_mat)

        # Calculate loss
        u = y_test[mask] - t_pred
        result += (u * (alpha - (u < 0).astype(int))).sum()

    return ("pinball", result, False)


# def objective(trial, data: lgb.Dataset):
def objective(
    trial,
    data: lgb.Dataset,
    n_val_pts: int,
    helper_mat: np.ndarray,
    val_helper_mat: np.ndarray,
):
    # Get datasets out
    X = data.get_data()
    y = data.get_label()

    # Split train test - Take last n_val_pts from each alpha
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    alphas = X[:, -1]
    test_mask = np.zeros(X.shape[0], dtype=bool)
    tmp = np.zeros((alphas == alphas[0]).sum(), dtype=bool)
    tmp[-n_val_pts:] = True
    for alpha in np.unique(alphas):
        test_mask[alphas == alpha] = tmp
    X_train = X[~test_mask]
    X_test = X[test_mask]
    y_train = y[~test_mask]
    y_test = y[test_mask]

    # Redefine lgb datasets
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, free_raw_data=False).construct()

    param = {
        "verbosity": -1,
        "boosting_type": "gbdt",
        "metric": "pinball",
        "num_iterations": trial.suggest_int("num_iterations", 100, 100),
        # "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1e-2, log=True),
        # "early_stopping_round": trial.suggest_int("use_early_stopping", 0, 0)
        # * trial.suggest_int("early_stopping_round", 1, 100),
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
        # "mc": [0, -1, 1],
        "mc": [0 for _ in range(X.shape[1] - 1)] + [1],
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "pinball")
    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dtest],
        callbacks=[pruning_callback],
        # fobj=partial(check_loss_grad_hess, alphas=X_train[:, -1]),
        fobj=partial(traj_loss_grad_hess, alphas=X_train[:, -1], helper_mat=helper_mat),
        # feval=pinball_metric,
        feval=partial(traj_pinball_metric, helper_mat=val_helper_mat),
    )

    # Save booster
    trial.set_user_attr(key="best_booster", value=gbm)

    # Return performance
    preds = gbm.predict(X_test)
    _, perf, _ = traj_pinball_metric(preds, dtest, val_helper_mat)
    return perf


def save_best_model(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def _feature_transform(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    if isinstance(x, pd.DataFrame | pd.Series):
        x = x.to_numpy()
    return x
    d1 = np.zeros(x.shape[0])
    d1[1:] = x[1:, 0] - x[:-1, 0]
    return np.hstack((x, d1.reshape(-1, 1)))
    return np.hstack(
        (x, np.sin(x[:, 0]).reshape(-1, 1), np.cos(x[:, 0]).reshape(-1, 1))
    )


class QuantileRegressionSimulator:
    """Quantile Regression Simulator

    This class fits quantile regressors to all columns
    """

    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        # m_factor: float,
        alphas: list[float],
        n_diff: int,
        dt: float,
        freq: float,
        random_state: Union[int, None] = None,
        n_val_pts: Union[int, float] = 0.25,
    ):
        n_feat = x.shape[1]

        # Add features
        x = _feature_transform(x)

        # Smooth target and 2nd derivatives
        y_all = np.zeros((x.shape[0], n_feat))
        for i in range(n_feat):
            y_all[:, i] = smooth_series(x[:, i], dt=dt, freq=freq)

        # # Get bootstrapped rows
        # n = round(m_factor * x.shape[0])
        # idx = self.rng.choice(x.shape[0], size=n, replace=True)

        # # reindex, add alphas
        # self.x_train = x[idx, :]
        # self.y_train = y_all[idx, :]
        # alphas = self.rng.uniform(size=(n, 1))

        # Duplicate, add alphas
        all_alphas = np.repeat(alphas, repeats=x.shape[0]).reshape(-1, 1)
        self.x_train = np.tile(x, (len(alphas), 1))
        self.y_train = np.tile(y_all, (len(alphas), 1))
        self.x_train = np.hstack((self.x_train, all_alphas))

        # Lightgbm preps
        self.datasets = []
        for i in range(n_feat):
            self.datasets.append(
                lgb.Dataset(
                    data=self.x_train,
                    label=self.y_train[:, i],
                    free_raw_data=False,
                ).construct()
            )

        # Save params
        # self.sample = sample
        self.rng = np.random.default_rng(random_state)
        self.alphas = alphas
        self.n_diff = n_diff

        # Number of validation points
        if isinstance(n_val_pts, float):
            assert (
                n_val_pts <= 1.0
            ), f"n_val_pts must be between 0 and 1 if a float, not {n_val_pts}"
            self.n_val_pts = round(n_val_pts * x.shape[0])
        else:
            self.n_val_pts = n_val_pts
        self.n_trn_pts = x.shape[0] - self.n_val_pts

    def train(self) -> List[lgb.Booster]:
        self.models = []
        self.helper_mat = _create_helper_mat(self.n_trn_pts)
        self.val_helper_mat = _create_helper_mat(self.n_val_pts)
        for i in range(len(self.datasets)):
            print(f"Running hyperparameter tuning for model {i + 1}...")

            # Old way
            curr_obj = partial(
                objective,
                data=self.datasets[i],
                n_val_pts=self.n_val_pts,
                helper_mat=self.helper_mat,
                val_helper_mat=self.val_helper_mat
                # objective, data=self.datasets[i], n_val_pts=self.n_val_pts
            )
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
    ) -> np.ndarray:
        if isinstance(alpha, float):
            alpha = [alpha for _ in self.models]
        alpha = list(alpha)

        _pred = np.zeros(x.shape)

        # Feature engineer and clean
        x = _feature_transform(x)

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
        pred = self.predict(X0, runif)
        return pred[0, :]

    def simulate_path(
        self,
        X0: np.ndarray,
        steps: int,
        sample: bool = True,
        X0_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        self.sample = sample
        preds = np.zeros((steps, len(X0)))
        curr_pos = X0.copy()
        if X0_prev is not None:
            prev_pos = X0_prev.copy()

        for i in range(steps):
            preds[i, :] = curr_pos
            hold_for_prev_pos = curr_pos.copy()
            if self.n_diff == 0:
                curr_pos = self.sol_fn(curr_pos)
            elif self.n_diff == 1:
                curr_pos += self.sol_fn(curr_pos)
            elif self.n_diff == 2:
                der2_add = self.sol_fn(curr_pos)
                curr_pos += (curr_pos - prev_pos) + der2_add
                prev_pos = hold_for_prev_pos
        return preds
