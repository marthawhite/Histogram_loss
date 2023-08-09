"""Module for reading experiment results to CSV from a set of experiments."""


import pandas as pd
import os
import json


def dir_to_df(base_dir):
    """Create a dataframe from a directory of JSON files.
    
    JSON files should be named {loss}_{model}_{dataset}.json and should contain
    arrays with the results for each epoch for each metric.
    
    Params:
        base_dir - the directory path containing the JSON files
    
    Returns: the combined dataframe
    """
    dfs = []
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        with open(file_path, "r") as in_file:
            data = json.load(in_file)
        df = pd.DataFrame(data)
        loss, model, ds = file[:-5].split("_")
        df["variant"] = loss
        df["model"] = model
        df["dataset"] = ds
        dfs.append(df)
    main_df = pd.concat(dfs)
    return main_df


def main():
    """Read experiment results to CSV."""
    base_dir = os.path.join("data", "bins_results")
    df = dir_to_df(base_dir)
    df.to_csv("bins_results.csv")


if __name__ == "__main__":
    main()
