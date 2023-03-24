"""Utilities for cartpole simulations."""

import pickle as pkl
from functools import partial
from typing import Callable, Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy import ndarray
from pysindy import SINDy
from scipy.integrate import solve_ivp

__all__ = [
    "cartpole_fn",
    "Cartpole",
    "Problem",
    "ThinnedModel",
    "cache_model",
    "load_model",
]


# Useful for containing the problem statement ------------------------


class Cartpole(NamedTuple):
    f: Callable[[ndarray], ndarray]
    m: float = 1
    L: float = 1
    M: float = 1


def cartpole_fn(t: ndarray, y: ndarray, params: Cartpole) -> ndarray:
    g: Final = -9.81

    f, m, L, M = params
    phi, _, phi_dot, s_dot = y

    z = np.zeros_like(y)
    z[:2] = [phi_dot, s_dot]

    # Update phi-dot
    numer = -(
        (M + m) * g * np.sin(phi)
        + f(t) * L * np.cos(phi)
        + m * L**2 * np.sin(phi) * np.cos(phi) * phi_dot**2
    )
    denom = L**2 * (M + m - m * np.cos(phi) ** 2)
    z[2] = numer / denom

    # Update s-dot
    numer = (
        m * L**2 * np.sin(phi) * phi_dot**2
        + f(t) * L
        + m * g * np.sin(phi) * np.cos(phi)
    )
    denom = L * (M + m - m * np.cos(phi) ** 2)
    z[3] = numer / denom

    return z


class Problem:
    def __init__(
        self,
        p: Cartpole,
        x0: np.ndarray,
        ts: tuple[float, float],
        step: float = 0.001,
    ) -> None:
        self.pars = p
        self.init = x0
        self.time = ts
        self.step = step

        fn = partial(cartpole_fn, params=self.pars)
        self.soln = solve_ivp(fn, self.time, self.init, dense_output=True)

    def times(
        self,
        a: float | None = None,
        b: float | None = None,
        step: float | None = None,
    ) -> ndarray:
        a = a if a is not None else self.time[0]
        b = b if b is not None else self.time[1]
        step = step if step is not None else self.step
        return np.arange(a, b + step, step)

    def simy(
        self,
        a: float | None = None,
        b: float | None = None,
        step: float | None = None,
    ) -> ndarray:
        return self.soln.sol(self.times(a, b, step))

    def plot(
        self,
        a: float | None = None,
        b: float | None = None,
        step: float | None = None,
    ) -> Figure:
        t = self.times(a, b, step)
        fig, axes = plt.subplots(ncols=2, figsize=(7, 2.75))
        for i, (ax, label) in enumerate(zip(axes, (r"$\phi$", "$s$"))):
            ax.plot(t, self.soln.sol(t).T[:, i])
            ax.set_title(label)
        fig.tight_layout()
        return fig


# Store models for later retrieval -----------------------------------


class ThinnedModel(NamedTuple):
    features: list[str]
    coefficients: np.ndarray


def cache_model(model: SINDy, path: str) -> str:
    with open(path, "wb") as f:
        thinned = ThinnedModel(model.get_feature_names(), model.coefficients())
        pkl.dump(thinned, f)
    return path


def load_model(path: str) -> ThinnedModel:
    with open(path, "rb") as f:
        return pkl.load(f)
