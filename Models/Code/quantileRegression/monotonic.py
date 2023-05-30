"""Adapted from https://github.com/microsoft/LightGBM/issues/5727#issue-1589436779"""

from typing import List, Union, Dict, Any, Tuple
from functools import partial
from itertools import repeat, chain

import numpy as np
import lightgbm as lgb
import pandas as pd


def _grad_rho(u, alpha) -> np.ndarray:
    return -(alpha - (u < 0).astype(float))


def check_loss_grad_hess(
    y_pred: np.ndarray, dtrain: lgb.basic.Dataset, alphas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = dtrain.get_label()
    grad = _grad_rho(y_train - y_pred, alphas)
    hess = np.ones(y_train.shape)
    return grad, hess


def _alpha_validate(
    alphas: Union[List[float], float],
) -> List[float]:
    if isinstance(alphas, float):
        alphas = [alphas]
    return alphas


def _prepare_x(
    x: Union[pd.DataFrame, pd.Series, np.ndarray],
    alphas: List[float],
) -> pd.DataFrame:
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
        x = pd.DataFrame(x)
    assert "_tau" not in x.columns, "Column name '_tau' is not allowed."
    _alpha_repeat_count_list = [list(repeat(alpha, len(x))) for alpha in alphas]
    _alpha_repeat_list = list(chain.from_iterable(_alpha_repeat_count_list))
    _repeated_x = pd.concat([x] * len(alphas), axis=0)

    _repeated_x = _repeated_x.assign(
        _tau=_alpha_repeat_list,
    )
    return _repeated_x


def _feature_transform(x: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    if hasattr(x, "iloc"):
        x = x.to_numpy()
    return np.hstack(
        (x, np.sin(x[:, 0]).reshape(-1, 1), np.cos(x[:, 0]).reshape(-1, 1))
    )


class MonotonicQuantileRegressor:
    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        m_factor: float,
        sample: bool = False,
        random_state: Union[int, None] = None,
    ):
        # Add features; clean
        x = _feature_transform(x)
        if hasattr(y, "iloc"):
            y = y.to_numpy()

        # Get bootstrapped rows
        self.rng = np.random.default_rng(random_state)
        n = round(m_factor * x.shape[0])
        idx = self.rng.choice(x.shape[0], size=n, replace=True)

        # reindex, add alphas
        self.x_train = x[idx, :]
        self.y_train = y[idx]
        alphas = self.rng.uniform(size=(n, 1))
        self.x_train = np.hstack((self.x_train, alphas))

        # Lightgbm preps
        self.dataset = lgb.Dataset(data=self.x_train, label=self.y_train)
        self.fobj = partial(check_loss_grad_hess, alphas=alphas[:, 0])

        # Save params
        self.sample = sample
        self.m_factor = m_factor

    def train(self, params: Dict[str, Any]) -> lgb.basic.Booster:
        self._params = params.copy()
        if "monotone_constraints" in self._params:
            self._params["monotone_constraints"].append(1)
        else:
            mc = [0 for _ in range(self.x_train.shape[1])]
            mc[-1] = 1
            self._params.update({"monotone_constraints": mc})
        self.model = lgb.train(
            train_set=self.dataset,
            verbose_eval=False,
            params=self._params,
            fobj=self.fobj,
            # feval=self.feval,
        )
        return self.model

    def predict(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ) -> np.ndarray:
        alphas = _alpha_validate(alphas)

        # Feature engineer
        x = _feature_transform(x)

        _x = _prepare_x(x, alphas)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x)).T
        return _pred

    def sol_fn(self, t: float, X0: np.ndarray):
        if self.sample:
            runif = self.rng.uniform()
        else:
            runif = 0.5

        if len(X0.shape) == 1:
            X0 = X0.reshape(1, -1)

        return [X0[0, 1], self.predict(X0, [runif])[0, 0]]
