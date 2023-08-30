"""Module to collect the results of an Atari experiment into a CSV file.

Results should be collected in a directory with a subdirectory for the results of each game.
"""

import os
import json
import pandas as pd


def read_json(filepath):
    """Read the json file to a pandas dataframe.
    
    Params:
        filepath - the path to the JSON file
    
    Returns: a pandas dataframe with keys as columns, values as series
    """
    with open(filepath, "r") as in_file:
        json_data = json.load(in_file)
    for k, v in json_data.items():
        json_data[k] = pd.Series(v)
    return pd.DataFrame.from_dict(json_data)


def read_game(data_dir, game):
    """Return a dataframe for the results from a single game.
    
    Params:
        data_dir - the path to the results directory
        game - the name of the game

    Returns: a pandas dataframe with the results, model, and game name
    """
    path = os.path.join(data_dir, game)
    models = ["hlg", "reg"]
    dfs = []
    for model in models:
        filename = f"{model}.json"
        filepath = os.path.join(path, filename)
        df = read_json(filepath)
        df["model"] = model
        dfs.append(df)
    full_df = pd.concat(dfs)
    full_df["game"] = game
    return full_df


def read_results(data_dir):
    """Create a dataframe with the results from multiple games.
    
    Params:
        data_dir - the path to the results directory
            contains subdirectories named by game with results
    
    Returns: a pandas dataframe with the results from each game
    """
    dfs = []
    for game in os.listdir(data_dir):
        dfs.append(read_game(data_dir, game))
    return pd.concat(dfs)


def main():
    """Convert atari results to csv."""
    data_dir = os.path.join("data", "atari_results")
    out_file = "atari_results.csv"
    df = read_results(data_dir)
    df.to_csv(out_file)


if __name__ == "__main__":
    main()
