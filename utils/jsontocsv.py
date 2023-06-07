import pandas as pd
import json

in_file = "log_test5.json"
out_file = "log_test5.csv"

with open(in_file, "r") as json_file:
    data = json.load(json_file)

models = []
for model, model_d in data.items():
    dfs = []
    for trial, trial_d in model_d.items():
        df = pd.DataFrame.from_dict(trial_d["results"])
        for hyper, hyper_val in trial_d["hypers"].items():
            df[hyper] = hyper_val
        df["trial"] = trial
        dfs.append(df)
    new_df = pd.concat(dfs)
    new_df["model"] = model
    models.append(new_df)
final = pd.concat(models)
final["epoch"] = final.index + 1

first_cols = ["model", "trial", "epoch"]
last_cols = [col for col in final.columns if col not in first_cols]
final = final[first_cols + last_cols]

final.to_csv(out_file, index=False)