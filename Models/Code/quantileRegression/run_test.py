from pathlib import Path
import json

import pickle
import sys
from typing import Any
import pandas as pd
import numpy as np

from data_gen import gen_cartpole_data
from qr_simulation import QuantileRegressionSimulator
from utils import make_serializable


datapath = Path.home() / "Box/NASA_Figures/data"

model = QuantileRegressionSimulator

params: dict[str, Any] = {
    "name": "qr_test38",  # Custom unique name used for saving predictions, parameters
    "model_name": model.__name__,
    "dt": 0.01,
    "model_params": {
        "m_factor": 100,
        "random_state": 5,
        "n_diff": 2,
        "freq": 1 / 4,
        "alpha_dist": "beta",
        "alpha_dist_params": {
            "a": 0.25,
            "b": 0.25,
        },
        "model_params": {
            "num_iterations": 500,
            "learning_rate": 1e-2,
        },
        "smooth_derv_est": True,
        "smoothing_samples": 100,
        "smoothing_perc": 0.95,
        "smoother": "meandiff",
    },
    "data_gen_params": {
        "de_params": {
            "mc": 1.5,
            "mp": 1.0,
            "len": 1.0,
            "mu_c": 0.05,
            "mu_p": 0.05,
            "k": 1 / 3,
            "f": lambda x: 0,
        },
        "tspan": (0, 50),
        "X0_gen": "normal",
        "X0_params": [
            {"loc": 0, "scale": 1},  # theta
            {"loc": 0, "scale": 3},  # theta_d
            {"loc": 0, "scale": 0},  # x
            {"loc": 0, "scale": 1},  # x_d
        ],
        "X0_num": 10,
        "noise_gen": "normal",
        "noise_params": {"scale": 0.1},
    },
    "train_num": 8,  # Use first n datasets for training
    "train_cutoff": 0.75,  # Use first __% of data to train
    "valid_num": None,
    "valid_cont": 2,  # Eval first __ sets from train_cutoff to the end
    "valid_train": 2,  # Eval first __ sets from start to finish
    "n_sims": 100,
    "var_names": ["theta", "x", "theta_d", "x_d"],
    "random_state": 6,
}

if __name__ == "__main__":
    # Check if overwriting
    if (datapath / f"validation/predictions/{params['name']}.csv").exists():
        resp = (
            input(
                f"Predictions have already been saved with name {params['name']}. Do"
                " you want to overwrite them ([y]/n)? "
            )
            + " "
        )
        if resp.lower()[0] == "n":
            sys.exit()

    # Propogate param
    params["model_params"]["dt"] = params["dt"]
    params["data_gen_params"]["dt"] = params["dt"]
    params["data_gen_params"]["random_state"] = params["random_state"]

    # Generate X0s
    rng = np.random.default_rng(params["random_state"])
    dp = params["data_gen_params"]
    params["data_gen_params"]["X0"] = list(
        zip(
            *[
                getattr(rng, dp["X0_gen"])(size=dp["X0_num"], **dp["X0_params"][j])
                for j in range(len(dp["X0_params"]))
            ]
        )
    )

    # Generate data
    data, true_data = gen_cartpole_data(**params["data_gen_params"])

    # Calculate validation sets
    params["valid_sets"] = {}
    for i in range(params["data_gen_params"]["X0_num"]):
        tmp = []
        if i < params["valid_train"]:
            tmp.append(1)
        if i < params["valid_cont"]:
            tmp.append(1 - params["train_cutoff"])
        if i >= params["train_num"]:
            tmp.append(1)
        if len(tmp) > 0:
            params["valid_sets"][i] = tmp

    # Construct train and test sets
    train_sets, valid_sets, valid_starts, val_num, true_sets = [], [], [], [], []
    print("Constructing datasets")
    for i in data.set_id.unique():
        curr_data = data[data.set_id == i].copy()
        n = round(params["train_cutoff"] * curr_data.shape[0])

        # Add to train data
        # if i in params["train_sets"]:
        if i < params["train_num"]:
            train_sets.append(curr_data.iloc[:n])

        # Add to validation data
        if i in params["valid_sets"]:
            for p in params["valid_sets"][i]:
                # Add validation data
                n = round(p * curr_data.shape[0])
                val_num.append(n)
                # valid_sets.append(curr_data.iloc[-n:])
                valid_sets.append(curr_data.copy())

                # Add true starting point
                X0 = (
                    true_data.loc[true_data.set_id == i, params["var_names"]]
                    .iloc[-n]
                    .to_numpy()
                )
                X0[[2, 3]] *= params["model_params"]["dt"]  # derivatives rescaled w/run
                valid_starts.append(X0)

                # Add true data
                curr_true_data = true_data[true_data.set_id == i].drop(
                    columns=params["var_names"]
                )
                curr_true_data = curr_true_data.rename(
                    columns={"theta_dd": "theta", "x_dd": "x"}
                )
                true_sets.append(curr_true_data)

    # Model is trained on one dataset
    train_data = pd.concat(train_sets)

    # Train model
    print("Training Model")
    curr_model = model(
        x=train_data[["theta", "x"]], seq_id=train_data.set_id, **params["model_params"]
    )
    curr_model.train()

    # Simulate over validation segments
    sim_data_list = []
    for i in range(len(valid_sets)):
        print(f"Generating simulations {i + 1} / {len(valid_sets)}")
        X0 = valid_starts[i]
        valid = valid_sets[i]
        curr_true_data = true_sets[i]
        n = val_num[i]

        # Simulate trajectories
        preds = curr_model.simulate_paths(
            X0=X0, n=params["n_sims"], steps=n
        )  # n_sims by n by n_feat

        # Convert to df and save
        curr_sims = pd.DataFrame(
            preds.reshape(-1, preds.shape[-1]), columns=["theta", "x"]
        )
        curr_sims["sim_id"] = np.repeat(range(params["n_sims"]), n)
        curr_sims["valid_id"] = i
        curr_sims["sim"] = True
        curr_sims["t"] = np.tile(valid.iloc[-n:]["t"].to_numpy(), params["n_sims"])
        curr_sims["set_id"] = valid["set_id"].iloc[0]
        curr_sims["true_data"] = False
        # for j in range(params["n_sims"]):
        #     curr_sims = pd.DataFrame(
        #         curr_model.simulate_path(X0=X0, steps=n),
        #         columns=["theta", "x"],
        #     )
        #     curr_sims["sim_id"] = j
        #     curr_sims["valid_id"] = i
        #     curr_sims["sim"] = True
        #     curr_sims["t"] = valid.iloc[-n:]["t"].to_numpy()
        #     curr_sims["set_id"] = valid["set_id"].iloc[0]
        #     curr_sims["true_data"] = False
        #     sim_data_list.append(curr_sims)
        sim_data_list.append(curr_sims)
        sim_data_list.append(valid.assign(valid_id=i, sim=False, true_data=False))
        sim_data_list.append(
            curr_true_data.assign(valid_id=i, sim=False, true_data=True)
        )

    sim_data = pd.concat(sim_data_list)

    # Add whether data was trained on
    sim_data = sim_data.merge(
        train_data[["t", "set_id"]], on=["t", "set_id"], how="left", indicator="train"
    )
    sim_data["train"] = sim_data["train"] == "both"

    print("Saving predictions, parameters, and model")

    # Save predictions
    sim_data.to_csv(
        datapath / f"validation/predictions/{params['name']}.csv", index=False
    )

    # Save parameters
    serializable_params = make_serializable(params)
    with open(datapath / f"validation/parameters/{params['name']}.json", "w") as f:
        f.write(json.dumps(serializable_params, indent=4))

    # Save model
    del curr_model.datasets
    with open(
        datapath / f"validation/model_objects/{params['name']}.pkl", "wb"
    ) as outp:
        pickle.dump(curr_model, outp, pickle.HIGHEST_PROTOCOL)
