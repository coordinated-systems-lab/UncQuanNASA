from pathlib import Path
import json

import pickle
import sys
from typing import Any
import pandas as pd
import numpy as np

from qr_simulation import QuantileRegressionSimulator
from utils import make_serializable


outpath = Path.home() / "Box/NASA_Figures/data"
inpath = Path.cwd() / "../../../Data/cartpoleData"
evalpath = Path.cwd() / "../../../Results/evaluation/predictions"

model = QuantileRegressionSimulator

params: dict[str, Any] = {
    "name": "qr_preds22",  # Custom unique name used for saving predictions, parameters
    "model_name": model.__name__,
    "model_params": {
        "m_factor": 10,
        "freq": 1 / 4,
        "alpha_dist": "beta",
        "alpha_dist_params": {
            "a": 0.25,
            "b": 0.25,
        },
        "dt": 0.01,
        "model_params": {
            "num_iterations": 2000,
            "learning_rate": 1e-3,
        },
        "smooth_derv_est": True,
        "smoothing_samples": None,
        "smoothing_perc": 1.0,
        "smoother": "meandiff",
        "smooth_window_size": 5,
        "convert_theta": True,
    },
    # Which datasets
    "datasets": [
        "det",
        "low_noise",
        "high_noise",
    ],  # det, low_noise, high_noise only options
    # Validation parameters
    # Remainder of train always validated (unless train_seconds == 400)
    # Others must be specified
    "valid_train": True,
    "valid_valid": True,
    "valid_test": True,
    "train_seconds": 400,  # Use first __ seconds of data to train
    "val_train_start": 400,  # Start val after __ seconds, same as train_seconds if None
    "val_train_seconds": 0,  # Use __ seconds of remaining data to val, None = all
    "n_sims": 100,
    "levels": [50, 80, 95],
    "var_names": ["theta", "x", "theta_d", "x_d"],
    "eval_modes": ["single", "multi"],
    "random_state": 7,
}

if __name__ == "__main__":
    # Check if overwriting
    if (outpath / f"validation/predictions/{params['name']}.csv").exists():
        resp = (
            input(
                f"Predictions have already been saved with name {params['name']}. Do"
                " you want to overwrite them ([y]/n)? "
            )
            + " "
        )
        if resp.lower()[0] == "n":
            sys.exit()

    # Propogate param, set params
    params["model_params"]["random_state"] = params["random_state"]

    # Loop through datasets
    all_sim_data = []
    for dname in params["datasets"]:
        # Get all relevent datasets
        valid_sets, valid_starts = {}, {}
        data = pd.read_csv(inpath / f"{dname}_train.csv", index_col="t")
        train = data.loc[: params["train_seconds"]].copy()

        # Get validation sets
        if params["val_train_seconds"] is None:
            params["val_train_seconds"] = 500 - params["train_seconds"]

        if params["val_train_seconds"] > 0:
            sp = (
                params["train_seconds"]
                if params["val_train_start"] is None
                else params["val_train_start"]
            )

            # Save validation data
            valid_sets["val_train"] = data.loc[
                sp : (sp + params["val_train_seconds"])
            ].copy()

            # Get correct starting point
            skiprows = round(sp / params["model_params"]["dt"])
            valid_starts["val_train"] = (
                pd.read_csv(
                    inpath / "det_train.csv",
                    nrows=1,
                    skiprows=range(1, skiprows + 1),
                )[params["var_names"]]
                .iloc[0]
                .to_numpy()
            )
        if params["valid_train"]:
            valid_sets["train"] = train.copy()
            valid_starts["train"] = (
                pd.read_csv(
                    inpath / "det_train.csv",
                    nrows=1,
                )[params["var_names"]]
                .iloc[0]
                .to_numpy()
            )
        if params["valid_valid"]:
            valid_sets["val"] = pd.read_csv(inpath / f"{dname}_val.csv", index_col="t")
            valid_starts["val"] = (
                pd.read_csv(
                    inpath / "det_val.csv",
                    nrows=1,
                )[params["var_names"]]
                .iloc[0]
                .to_numpy()
            )
        if params["valid_test"]:
            valid_sets["test"] = pd.read_csv(
                inpath / f"{dname}_test.csv", index_col="t"
            )
            valid_starts["test"] = (
                pd.read_csv(
                    inpath / "det_test.csv",
                    nrows=1,
                )[params["var_names"]]
                .iloc[0]
                .to_numpy()
            )

        # Add valid starts to parameters
        params["valid_starts"] = {k: list(v) for k, v in valid_starts.items()}

        # Train model
        print("Training Model")
        curr_model = model(x=train, **params["model_params"])
        curr_model.train()

        # Levels to quantiles
        alpha = [(1 - lev / 100) / 2 for lev in params["levels"]]
        q = alpha + [1 - a for a in alpha]
        q_names = [
            f"{pref}_{lev}" for pref in ["lower", "upper"] for lev in params["levels"]
        ]

        # Simulate over validation segments
        sim_data_list = []
        for name, val_data in valid_sets.items():
            for eval_mode in params["eval_modes"]:
                if eval_mode == "multi":
                    # Simulate trajectories
                    sims = curr_model.simulate_paths(
                        valid_starts[name],
                        force=val_data.force_in.to_numpy(),
                        n=params["n_sims"],
                        steps=val_data.shape[0],
                    )  # nsims x nsteps x 4

                    for i, var in enumerate(params["var_names"]):
                        # Caculate quantiles
                        sim_df = pd.DataFrame(
                            np.quantile(sims[..., i], axis=0, q=q).T,
                            columns=q_names,
                            index=val_data.index,
                        )
                        sim_df["mean"] = sims[..., i].mean(axis=0)
                        sim_df["actual"] = val_data[var]
                        sim_df["name"] = name
                        sim_df["variable"] = var
                        sim_df["t"] = sim_df.index
                        sim_df["eval_mode"] = eval_mode
                        sim_df = sim_df.reset_index(drop=True)
                        sim_data_list.append(sim_df)
                elif eval_mode == "single":
                    sims = curr_model.predict_single(val_data, levels=params["levels"])
                    for var in params["var_names"]:
                        sim_df = sims[sims.variable == var].copy()
                        sim_df.index = val_data.index
                        sim_df["actual"] = val_data[var]
                        sim_df["name"] = name
                        sim_df["t"] = sim_df.index
                        sim_df["eval_mode"] = eval_mode
                        sim_df = sim_df.reset_index(drop=True)
                        sim_data_list.append(sim_df)

        sim_data = pd.concat(sim_data_list)
        sim_data["noise"] = dname

        print("Saving predictions, parameters, and model")

        # Store predictions
        all_sim_data.append(sim_data)

        # Save model
        del curr_model.datasets
        with open(
            outpath / f"validation/model_objects/{params['name']}_{dname}.pkl", "wb"
        ) as outp:
            pickle.dump(curr_model, outp, pickle.HIGHEST_PROTOCOL)

    # Save predictions
    all_data = pd.concat(all_sim_data)
    all_data.to_csv(
        outpath / f"validation/predictions/{params['name']}.csv", index=False
    )

    # Save to experiment eval directory as well
    all_data[
        (all_data.t - all_data.groupby("name")["t"].transform("min")) <= 10
    ].to_csv(evalpath / f"{params['name']}.csv", index=False)

    # Save parameters
    serializable_params = make_serializable(params)
    with open(outpath / f"validation/parameters/{params['name']}.json", "w") as f:
        f.write(json.dumps(serializable_params, indent=4))
