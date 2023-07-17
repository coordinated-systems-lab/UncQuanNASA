import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from typing import Any


def cartpole(t: float, u: list[float], p: dict[str, Any]):
    # u = [theta, theta_d, x, x_d]

    # Parameters
    g = 9.8
    du = np.zeros(4)

    # theta_d
    du[0] = u[1]

    # theta_dd
    du[1] = (
        (p["mc"] + p["mp"]) * g * np.sin(u[0])
        - np.cos(u[0])
        * (
            p["f"](t)
            + p["mp"] * p["len"] * (u[1] ** 2) * np.sin(u[0])
            - p["mu_c"] * u[3]
        )
        - ((p["mc"] + p["mp"]) * p["mu_p"] * u[1]) / (p["mp"] * p["len"])
    ) / (
        (p["mc"] + p["mp"]) * (p["k"] + 1) * p["len"]
        - p["mp"] * p["len"] * (np.cos(u[0])) ** 2
    )

    # x_d
    du[2] = u[3]

    # x_dd
    du[3] = (
        p["f"](t)
        - p["mp"] * p["len"] * (du[1] * np.cos(u[0]) - (u[1] ** 2) * np.sin(u[0]))
        - p["mu_c"] * u[3]
    ) / (p["mc"] + p["mp"])

    return du


def _check_list(arg, max_len=1) -> list:
    if not isinstance(arg, list):
        arg = [arg for _ in range(max_len)]
    if len(arg) != max_len:
        if len(arg) == 1:
            arg = [arg[0] for _ in range(max_len)]
        else:
            raise ValueError(f"Incompatible length {len(arg)}; max len is {max_len}")
    return arg


def gen_cartpole_data(
    de_params: list[dict] | dict,
    tspan: list[tuple[float, float]] | tuple[float, float],
    dt: list[float] | float,
    X0: list[tuple] | tuple,
    noise_gen: list[str] | str,
    noise_params: list[dict] | dict,
    random_state: int | None = None,
    **kwargs,
):
    # Preprocessing
    try:
        max_len = max(
            len(arg)
            for arg in [de_params, tspan, dt, X0, noise_gen, noise_params]
            if isinstance(arg, list)
        )
    except ValueError:
        max_len = 1

    de_params = _check_list(de_params, max_len=max_len)
    tspan = _check_list(tspan, max_len=max_len)
    dt = _check_list(dt, max_len=max_len)
    X0 = _check_list(X0, max_len=max_len)
    noise_gen = _check_list(noise_gen, max_len=max_len)
    noise_params = _check_list(noise_params, max_len=max_len)

    # Prepare for dataset creation
    rng = np.random.default_rng(random_state)
    data_list, true_data_list = [], []
    nsets = len(de_params)

    # Do it
    for i in range(nsets):
        t = np.linspace(*tspan[i], round(tspan[i][1] / dt[i]))

        # Solve IVP
        res = solve_ivp(
            cartpole, tspan[i], X0[i], args=[de_params[i]], dense_output=True
        )
        sol_data = res.sol(t).T

        # Add noise
        data = sol_data + getattr(rng, noise_gen[i])(
            size=sol_data.shape, **noise_params[i]
        )
        data = pd.DataFrame(data[:, [0, 2]], columns=["theta", "x"])
        true_data = pd.DataFrame(sol_data, columns=["theta", "theta_d", "x", "x_d"])
        data["t"], true_data["t"] = t, t
        data["set_id"], true_data["set_id"] = i, i
        data_list.append(data)
        true_data_list.append(true_data)

    # Add true second derivative to true data
    true_data = pd.concat(true_data_list).reset_index(drop=True)
    true_data_d = pd.DataFrame(
        [
            cartpole(
                row.t,
                [row.theta, row.theta_d, row.x, row.x_d],
                p=de_params[int(row.set_id)],
            )
            for _, row in true_data.iterrows()
        ],
        columns=["theta_d", "theta_dd", "x_d", "x_dd"],
    )
    true_data = pd.concat([true_data, true_data_d[["theta_dd", "x_dd"]]], axis=1)

    return pd.concat(data_list), true_data
