from pathlib import Path
import json

# import pickle
import sys
from typing import Any
import pandas as pd
from data_gen import gen_cartpole_data
from qr_simulation import QuantileRegressionSimulator
from utils import make_serializable


datapath = Path.home() / "Box/NASA_Figures/data"

model = QuantileRegressionSimulator

params: dict[str, Any] = {
    "name": "qr_test6",  # Custom unique name used for saving predictions, parameters
    "model_name": model.__name__,
    "model_params": {
        "m_factor": 100,
        "random_state": 5,
        "n_diff": 2,
        "freq": 1 / 3,
        "dt": 0.01,
        "model_params": {
            "num_iterations": 1000,
            "learning_rate": 1e-3,
        },
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
        "dt": 0.01,
        "X0": [
            (0.524, 0.0, 0.0, 0.5),
            (-1, 2.0, 0.0, 0.0),
            (0.0, -0.5, 0.0, -0.5),
        ],
        "noise_gen": "normal",
        "noise_params": {"scale": 0.1},
    },
    "train_sets": {
        0: 0.75,
        1: 0.75,
        # 2: 0.75,
    },
    "valid_sets": {
        0: [1, 0.25],
        1: [1, 0.25],
        2: [1],
    },
    "n_sims": 100,
    "var_names": ["theta", "x", "theta_d", "x_d"],
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

    # Generate data
    data, true_data = gen_cartpole_data(**params["data_gen_params"])

    # Construct train and test sets
    train_sets, valid_sets, valid_starts, val_num = [], [], [], []
    print("Constructing datasets")
    for i in data.set_id.unique():
        curr_data = data[data.set_id == i].copy()

        # Add to train data
        if i in params["train_sets"]:
            n = round(params["train_sets"][i] * curr_data.shape[0])
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

    # Model is trained on one dataset
    train_data = pd.concat(train_sets)

    # Train model
    print("Training Model")
    curr_model = model(x=train_data[["theta", "x"]], **params["model_params"])
    curr_model.train()

    # Simulate over validation segments
    sim_data_list = []
    for i in range(len(valid_sets)):
        print(f"Generating simulations {i + 1} / {len(valid_sets)}")
        X0 = valid_starts[i]
        valid = valid_sets[i]
        n = val_num[i]

        # Simulate trajectories
        for j in range(params["n_sims"]):
            curr_sims = pd.DataFrame(
                curr_model.simulate_path(X0=X0, steps=n),
                columns=["theta", "x"],
            )
            curr_sims["sim_id"] = j
            curr_sims["valid_id"] = i
            curr_sims["sim"] = True
            curr_sims["t"] = valid.iloc[-n:]["t"].to_numpy()
            curr_sims["set_id"] = valid["set_id"].iloc[0]
            sim_data_list.append(curr_sims)
        sim_data_list.append(valid.assign(valid_id=i, sim=False))
    sim_data = pd.concat(sim_data_list)

    # Save predictions
    sim_data.to_csv(
        datapath / f"validation/predictions/{params['name']}.csv", index=False
    )

    # Save parameters
    serializable_params = make_serializable(params)
    with open(datapath / f"validation/parameters/{params['name']}.json", "w") as f:
        f.write(json.dumps(serializable_params, indent=4))

    # # Save model
    # with open(
    #     datapath / f"validation/model_objects/{params['name']}.pkl", "wb"
    # ) as outp:
    #     pickle.dump(curr_model, outp, pickle.HIGHEST_PROTOCOL)
