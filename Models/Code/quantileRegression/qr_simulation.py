"""Adapted from https://github.com/microsoft/LightGBM/issues/5727#issue-1589436779"""

from typing import Any, List, Union, Tuple
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
) -> np.ndarray:
    x_hat, _ = getattr(pynumdiff.smooth_finite_difference, smoother)(
        x, dt, params, options={"iterate": True}
    )

    return np.column_stack((x_hat[:-1], x_hat[1:] - x_hat[:-1]))


def sample_smooth_series(
    x: np.ndarray,
    dt: float,
    params: list,
    rng: np.random.Generator,
    n_samples: int,
    perc_sample: float,
    smoother: str,
) -> np.ndarray:
    n = (len(x) - 1) * n_samples
    xhat = np.zeros((n, 2))

    # Average differences
    diff_std = (x[1:] - x[:-1]).std()  # / 3

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
        ind = np.arange(i * (len(x) - 1), (i + 1) * (len(x) - 1), dtype=int)
        xhat[ind, :] = smooth_series(curr_x, dt, params, smoother)
    return xhat


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    x1 = x[1:] - x[:-1]

    return np.column_stack((x[:-1], x1))


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.Dataset, alphas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = dtrain.get_label()
    grad = (y_pred > y_train).astype(int) - alphas
    hess = np.ones(y_train.shape)
    return grad, hess


def _check_x(
    x: Union[pd.DataFrame, pd.Series, np.ndarray], feature_names: list[str]
) -> np.ndarray:
    # Convert type to numpy
    if isinstance(x, pd.DataFrame | pd.Series):
        x = x.reindex(columns=feature_names).to_numpy()

    return x


def _feature_transform(
    x: np.ndarray, feature_names: list, convert_theta: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if convert_theta:
        # Convert theta
        idx = feature_names.index("theta")
        x[:, idx] = np.mod(x[:, idx], 2 * np.pi)

    # x to constant
    keep_feats = ["theta", "theta_d", "x_d", "force_in", "alpha"]
    return x, np.array([i for i, f in enumerate(feature_names) if f in keep_feats])


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
        dt: float,
        freq: float,
        model_params: dict[str, Any],
        alpha_dist: str = "uniform",
        smooth_derv_est: bool = True,
        smoothing_samples: int | None = None,
        smoothing_perc: float = 0.8,
        alpha_dist_params: dict | None = None,
        random_state: Union[int, None] = None,
        smoother: str = "butterdiff",
        convert_theta: bool = True,
    ):
        self.nrow = x.shape[0]
        self.n_feat = 2
        self.rng = np.random.default_rng(random_state)

        # Add features
        self.feature_names_ = ["theta", "x", "theta_d", "x_d", "force_in"]
        x = _check_x(x, self.feature_names_)

        # Smooth target and 2nd derivatives
        force = x[:, self.feature_names_.index("force_in")][:-1]
        if smoothing_samples is None or not smooth_derv_est:
            n = self.nrow - 1
        else:
            n = (self.nrow - 1) * smoothing_samples
            force = np.tile(force, reps=smoothing_samples)

        xhat = np.zeros((3, n, self.n_feat))
        self.resids = np.zeros((2, self.nrow, self.n_feat))
        for feat in ["theta", "x"]:
            feat_idx = self.feature_names_.index(feat)

            for i, suf in enumerate(["", "_d"]):
                idx = self.feature_names_.index(f"{feat}{suf}")

                if smooth_derv_est:
                    sp = _est_smoothing_params(
                        x[:, idx], dt=dt, freq=freq, smoother=smoother
                    )
                    if smoothing_samples is None:
                        xhat[i : (i + 2), :, feat_idx] = smooth_series(
                            x[:, idx],
                            dt=dt,
                            params=sp,
                            smoother=smoother,
                        ).T
                    else:
                        xhat[i : (i + 2), :, feat_idx] = sample_smooth_series(
                            x[:, idx],
                            dt=dt,
                            params=sp,
                            rng=self.rng,
                            n_samples=smoothing_samples,
                            perc_sample=smoothing_perc,
                            smoother=smoother,
                        ).T
                else:
                    xhat[i : (i + 2), :, feat_idx] = finite_diff(x[:, idx], dt=dt).T

                # Add resids
                if n == self.nrow:
                    self.resids[i, :, feat_idx] = x[:, idx] - xhat[i, :, feat_idx]

        self.x = np.hstack((xhat[0, ...], xhat[1, ...], force.reshape(-1, 1)))
        x = self.x
        self.d2 = xhat[2, ...]

        # Get bootstrapped rows
        n = round(m_factor * self.nrow)
        bs_idx = self.rng.choice(x.shape[0], size=n, replace=True)

        # reindex
        self.x_train = x[bs_idx, :]
        self.y_train = self.d2[bs_idx, :]

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
        self.dt = dt
        self.freq = freq
        self.smoother = smoother
        self.model_params = model_params
        self.convert_theta = convert_theta

    def train(self) -> List[lgb.Booster]:
        self.models = []
        for i in range(len(self.datasets)):
            # Get datasets out
            X = self.datasets[i].get_data()
            y = self.datasets[i].get_label()

            # feature transform
            X, f_idx = _feature_transform(
                X, self.feature_names_ + ["alpha"], convert_theta=self.convert_theta
            )

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

    def simulate_paths(
        self,
        X0: np.ndarray,
        force: np.ndarray,
        n: int,
        steps: int,
        sample: bool = True,
    ) -> np.ndarray:
        self.sample = sample

        # Instantiate array to hold predictions
        preds = np.zeros((n, steps, 2 * self.n_feat))

        # Fill in n starting positions
        curr_pos = np.zeros((n, len(X0)))
        curr_pos[:] = X0
        der2 = np.zeros((n, self.n_feat))

        for i in range(steps):
            preds[:, i, :] = curr_pos.copy()

            # Add alpha and force
            x = np.hstack(
                (
                    curr_pos,
                    np.repeat(force[i], repeats=n).reshape(-1, 1),
                    self.rng.uniform(size=(n, 1)),
                )
            )

            # Feature transform
            x, f_idx = _feature_transform(
                x, self.feature_names_ + ["alpha"], convert_theta=self.convert_theta
            )

            # Make 2nd derivative predictions
            for i in range(self.n_feat):
                der2[:, i] = self.models[i].predict(x[:, f_idx])

            # Propogate
            curr_pos[:, self.n_feat :] += der2
            curr_pos[:, : self.n_feat] += curr_pos[:, self.n_feat :] * self.dt

        # Add back in noise
        for i in range(2):
            for j in range(self.n_feat):
                preds[:, :, 2 * i + j] += self.rng.choice(
                    self.resids[i, :, j], size=(n, steps), replace=True
                )

        return preds

    def predict_single(self, X: pd.DataFrame, levels=list[int]) -> pd.DataFrame:
        # Levels to quantiles
        alpha = [(1 - lev / 100) / 2 for lev in levels]
        q = alpha + [1 - a for a in alpha]
        q_names = [f"{pref}_{lev}" for pref in ["lower", "upper"] for lev in levels]

        # Convert X
        x = _check_x(X, self.feature_names_)

        # Make predictions
        res_list = []
        for q_, q_name in zip(q, q_names):
            x_ = np.hstack((x, np.repeat(q_, x.shape[0]).reshape(-1, 1)))

            # feature transform
            x_, f_idx = _feature_transform(
                x_, self.feature_names_ + ["alpha"], convert_theta=self.convert_theta
            )

            for i, model in enumerate(self.models):
                var = ["theta", "x"][i]
                tmp = pd.DataFrame()

                # Make 2nd derivative predictions
                p = model.predict(x_[:, f_idx])
                tmp["pred"] = p
                tmp["variable"] = f"{var}_d"
                tmp["quantile"] = q_name
                tmp["t"] = X.index
                res_list.append(tmp.copy())

                # Make/propogate first derivative predictions?
                tmp["pred"] = self.dt * X[f"{var}_d"].to_numpy()
                tmp["variable"] = var
                tmp["quantile"] = q_name
                tmp["t"] = X.index
                res_list.append(tmp)
        all_preds = pd.concat(res_list)

        # Pivot wide
        all_preds = pd.pivot_table(
            all_preds, index=["variable", "t"], columns="quantile", values="pred"
        ).reset_index()

        return all_preds
