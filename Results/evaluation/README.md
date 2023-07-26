# Cartpole Experiment Evaluation

The purpose of the code in this section is to provide a consistent framework for evaluation of predictions produced by different methods.

## Evaluation Process

Here is the process:

1. Place predictions csv in the predictions folder; the name of the file is the name of the method (do not push data in this folder to git)
2. In a terminal, navigate to this directory
3. If the name of your file is "preds.csv", run `python evaluate.py preds.csv`. This will produce a corresponding file with the same name in the results folder. **Push the csv in the `results` folder to github but not the `predictions` or `full_results`, because of size restrictions**.
4. Use `prediction_comparison.ipynb` to compare predictions made by different methods

NOTE: I have put `qr_preds.csv` in all three folders in git to serve as an example/template.

## Format of Predictions csv

The predictions file must have a strict format, otherwise the evaluation will not work. It must have the following columns, named exactly the same as below:

* `name`
    * Defintion: Which eval task. All three should appear in the dataset for a fair evaluation.
    * Values: `test`, `train`, and `val`
* `variable`
    * Definition: The variable the current predictions are being made for
    * Values: `theta`, `theta_d`, `x`, `x_d`
* `t`
    * Definition: Time from start, in seconds. Only first ten seconds are used and needed, but more can be provided.
    * Values: float values
* `noise`
    * Definition: Which noise level is being used
    * Values: `det`, `low_noise`, `high_noise`
* `eval_mode`
    * Definition: Whether single or multi-step predictions are being evaluated
    * Values: `single`, `multi`
* `lower_95`: Lower bound of 95% prediction interval
* `lower_80`: Lower bound of 80% prediction interval
* `lower_50`: Lower bound of 50% prediction interval
* `upper_95`: Upper bound of 95% prediction interval
* `upper_80`: Upper bound of 80% prediction interval
* `upper_50`: Upper bound of 50% prediction interval

Once datasets have been filtered to first 10 seconds for each eval task, there should be exactly 72,072 rows:

`3 names * 4 variables * 1001 time steps (0-10 inclusive) * 3 noises * 2 eval_modes = 72,072`
