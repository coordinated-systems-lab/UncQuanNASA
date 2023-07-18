import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
import json
import inspect


def make_serializable(params: dict) -> dict:
    params = {**params}  # Make a copy
    for key in params:
        if type(params[key]) is dict:
            params[key] = make_serializable(params[key])
        else:
            try:
                json.dumps(params[key])
            except (TypeError, OverflowError):
                if key == "f":
                    funcString = str(inspect.getsourcelines(params[key])[0])
                    params[key] = funcString.strip("['\\n'],").split('"f": ')[1]
                else:
                    params[key] = params[key].__class__.__name__
    return params


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


def _add_second_derivative(df: pd.DataFrame, de_params: dict) -> pd.DataFrame:
    df_d = pd.DataFrame(
        [
            cartpole(
                idx,
                [row.theta, row.theta_d, row.x, row.x_d],
                p=de_params,
            )
            for idx, row in df.iterrows()
        ],
        columns=["theta_d", "theta_dd", "x_d", "x_dd"],
        index=df.index
    )

    return pd.concat([df, df_d[["theta_dd", "x_dd"]]], axis=1)


def gen_cartpole_data(
    de_params: dict,
    tspan: tuple[float, float],
    dt: float,
    X0: tuple,
    noise_gen: str,
    noise_params: dict,
    random_state: int | None = None,
):
    # Prepare for dataset creation
    rng = np.random.default_rng(random_state)

    # time variable
    t = np.linspace(*tspan, round(tspan[1] / dt) + 1)

    # Solve IVP
    res = solve_ivp(cartpole, tspan, X0, args=[de_params], dense_output=True)
    sol_data = res.sol(t).T

    # Add noise
    data = sol_data + getattr(rng, noise_gen)(size=sol_data.shape, **noise_params)
    data = pd.DataFrame(data, columns=["theta", "theta_d", "x", "x_d"])
    true_data = pd.DataFrame(sol_data, columns=["theta", "theta_d", "x", "x_d"])
    data.index, true_data.index = t, t
    data.index.name = "t"
    true_data.index.name = "t"

    # Add second derivatives
    data = _add_second_derivative(data, de_params=de_params)
    true_data = _add_second_derivative(true_data, de_params=de_params)

    # Add force
    data["force_in"] = de_params["f"](data.index)
    true_data["force_in"] = de_params["f"](true_data.index)

    return data, true_data


if __name__ == "__main__":
    # Set default params
    base_params = {
        "de_params": {
            "mc": 1.0,
            "mp": 0.1,
            "len": 0.5,
            "mu_c": 0.00005,
            "mu_p": 0.000002,
            "k": 1 / 3,
            "f": lambda x: 5 * np.sin(6 * x),
        },
        "tspan": (0, 500),
        "X0": (0.3, 1.0, 0.0, -0.72),  # theta, theta_d, x, x_d
        "dt": 0.01,
        "noise_gen": "normal",
    }

    # Set test params
    test_params = {
        "de_params": {
            "mc": 1.0,
            "mp": 0.1,
            "len": 0.5,
            "mu_c": 0.00005,
            "mu_p": 0.000002,
            "k": 1 / 3,
            "f": lambda x: -4 * np.cos(4.5 * x),
        },
        "tspan": (0, 200),
        "X0": (-0.1, -3.0, 0.0, 0.15),  # theta, theta_d, x, x_d
        "dt": 0.01,
        "noise_gen": "normal",
    }

    # Noise magnitudes
    noise_vals = {
        "det": 0.0,
        "low_noise": 0.05,
        "high_noise": 0.1,
    }

    for name, noise in noise_vals.items():
        # Simulate data
        params = base_params | {
            "noise_params": {"scale": noise},
        }
        data, _ = gen_cartpole_data(**params)

        # Save train and validation
        data.loc[:400].to_csv(f"../cartpoleData/{name}_train.csv")
        data.loc[400:].to_csv(f"../cartpoleData/{name}_val.csv")

        # Save parameters
        ser_params = make_serializable({"name": name} | params)
        with open(f"../dataParams/{name}.json", "w") as f:
            f.write(json.dumps(ser_params, indent=4))

        # Test data
        params = test_params | {
            "noise_params": {"scale": noise},
        }
        data, _ = gen_cartpole_data(**params)

        # Save train and validation
        data.to_csv(f"../cartpoleData/{name}_test.csv")

        # Save parameters
        ser_params = make_serializable({"name": name} | params)
        with open(f"../dataParams/{name}_test.json", "w") as f:
            f.write(json.dumps(ser_params, indent=4))
