import pandas as pd
from pathlib import Path
import sys


def get_interval_cols(cols: list[str]) -> dict[int, list[str]]:
    lower_cols = [col for col in cols if col.startswith("lower")]
    lower_nums = [int(col.split("_")[1]) for col in lower_cols]
    upper_cols = [col for col in cols if col.startswith("upper")]
    upper_nums = [int(col.split("_")[1]) for col in upper_cols]
    assert (
        lower_nums == upper_nums
    ), f"Incompatible columns {upper_cols = }; {lower_cols = }"

    return {num: [f"lower_{num}", f"upper_{num}"] for num in lower_nums}


def interval_score(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    col_map = get_interval_cols(list(df.columns))
    for num, (l, u) in col_map.items():
        m = 2 / (1 - num / 100)
        df[f"interval_{num}"] = df[u] - df[l]
        too_high = df.actual > df[u]
        too_low = df.actual < df[l]
        df.loc[too_high, f"interval_{num}"] += m * (
            df.loc[too_high, "actual"] - df.loc[too_high, u]
        )
        df.loc[too_low, f"interval_{num}"] += m * (
            df.loc[too_low, l] - df.loc[too_low, "actual"]
        )
    w_cols = [f"interval_{num}" for num in col_map]
    return df, w_cols


def eval_preds(preds: pd.DataFrame) -> pd.DataFrame:
    avg_dist = None

    # Get average distance from starting point
    avg_dist = (
        preds.query("noise == 'det'")
        .groupby(["variable", "name", "eval_mode"], as_index=False)["actual"]
        .apply(lambda x: (x - x.iloc[0]).abs().mean())
        .rename(columns={"actual": "avg_dist"})
    )

    # Evals and reshape
    preds, wcols = interval_score(preds)
    preds = pd.melt(
        preds,
        id_vars=["name", "variable", "noise", "t", "eval_mode"],
        value_vars=wcols,
        var_name="interval",
        value_name="score",
    )
    preds["interval"] = preds.interval.str.split("_").str[1].astype(int)

    # Join and account for average
    res = preds.merge(avg_dist, on=["variable", "eval_mode", "name"])
    res["norm_score"] = res.score / res.avg_dist

    return res


def check_data(df: pd.DataFrame) -> pd.DataFrame:
    col_vals = {
        "name": ["test", "train", "val"],
        "variable": ["theta", "theta_d", "x", "x_d"],
        "noise": ["det", "low_noise", "high_noise"],
        "eval_mode": ["single", "multi"],
        "t": None,
        "lower_95": None,
        "lower_80": None,
        "lower_50": None,
        "upper_95": None,
        "upper_80": None,
        "upper_50": None,
    }

    # Check values match
    for colname, vals in col_vals.items():
        # Assert column is in dataset
        assert colname in df.columns, f"predictions file missing {colname}"

        # Assert correct values if relevant
        if vals is not None:
            uniq_vals = df[colname].unique()
            assert set(vals) == set(
                uniq_vals
            ), f"{colname} should have values: {vals}; instead found {uniq_vals}"

    # Reset t, filter
    df["t"] -= df.groupby("name")["t"].transform("min")
    df = df.query("t <= 10")

    # Assert correct number of rows
    assert df.shape[0] == 72072, f"Dataset should have 72,072 rows, found {df.shape[0]}"

    # Drop extra columns
    extra_cols = [col for col in df.columns if col not in col_vals]
    print(f"Dropping extra columns: {extra_cols}")
    df = df.drop(columns=extra_cols)

    return df


def create_actuals_lookup(datapath: Path) -> pd.DataFrame:
    df_list = []
    var_names = ["theta", "theta_d", "x", "x_d"]
    for file in datapath.iterdir():
        for eval_mode in ["multi", "single"]:
            # Read in
            tmp = pd.read_csv(file, usecols=["t"] + var_names)

            # Target changes depending on objective
            if eval_mode == "single":
                tmp[var_names] = tmp[var_names].shift(-1) - tmp[var_names]

            # Reset time to 0; filter to first 10 seconds
            tmp["t"] -= tmp["t"].min()
            tmp = tmp.query("t <= 10")

            # Pivot long
            tmp = pd.melt(tmp, id_vars=["t"], var_name="variable", value_name="actual")

            # Extract eval task and noise level from name
            name = file.stem
            tmp["noise"] = "_".join(name.split("_")[:-1])
            tmp["name"] = name.split("_")[-1]
            tmp["eval_mode"] = eval_mode

            # Save to list
            df_list.append(tmp)
    return pd.concat(df_list, ignore_index=True)


if __name__ == "__main__":
    # Get filename from command line arg
    filename = f"predictions/{sys.argv[1]}"

    # Read in predictions
    preds = pd.read_csv(filename)

    print(preds.shape)
    sys.exit

    # Perform data checks
    preds = check_data(preds)

    # Add in actuals column
    datapath = Path(__file__).parents[2] / "Data/cartpoleData"
    actuals = create_actuals_lookup(datapath)
    preds = preds.merge(actuals, on=["name", "noise", "variable", "t", "eval_mode"])

    # Evaluate
    res = eval_preds(preds)

    # Aggregate and summarize
    res_summarized = res.groupby(["variable", "name", "eval_mode"], as_index=False)[
        "norm_score"
    ].mean()

    # Save full results
    res.to_csv(f"full_results/{sys.argv[1]}", index=False)

    # Save summarized results
    res_summarized.to_csv(f"results/{sys.argv[1]}", index=False)
