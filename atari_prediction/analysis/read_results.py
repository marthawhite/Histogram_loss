"""Read results from the Atari rerun experiment and save to CSV.

Saves both the individual epoch results as well as the overall test metrics.
"""

import os
import pandas as pd
import json


def get_games(path):
    """Return a list of games read from a file.
    
    Params:
        path - the path to the games file

    Returns: the list of games
    """
    with open(path, "r") as in_file:
        games = in_file.read().splitlines()
    return games


def read_full(base_dir, games):
    """Read the results for each epoch into a dataframe.
    
    Params:
        base_dir - the directory containing the results for each game
        games - the names of the Atari games
    
    Returns: the Pandas DataFrame with the results
    """
    dfs = []
    for game in games:
        for loss in ["hlg", "reg"]:
            filename = f"{loss}.json"
            res_path = os.path.join(base_dir, game, filename)
            with open(res_path, "r") as in_file:
                data = json.load(in_file)
            df = pd.DataFrame(data)
            df["model"] = loss
            df["game"] = game
            dfs.append(df)
    full = pd.concat(dfs)
    return full


def read_meta(base_dir, games):
    """Read the overall test metrics for each game.
    
    Params:
        base_dir - the directory containing the results for each game
        games - the names of the Atari games

    Returns: the Pandas DataFrame with the test results
    """
    keys = ["reg_mae", "reg_mse", "reg_loss", "hl_mae", "hl_mse", "hl_loss", "game"]
    results = {key: [] for key in keys}
    for game in games:
        file_path = os.path.join(base_dir, game, "results.json")
        with open(file_path, "r") as in_file:
            data = json.load(in_file)
        for key, val in data.items():
            results[key].append(val)
        results["game"].append(game)
    meta = pd.DataFrame(results)
    return meta


def main():
    """Read the Atari results and save to CSV files."""
    base_dir = os.path.join("data", "rerun_results")
    games_file = os.path.join("atari_prediction", "games.txt")
    full_file = "rerun_full.csv"
    meta_file = "rerun_meta.csv"

    games = get_games(games_file)

    full_df = read_full(base_dir, games)
    full_df.to_csv(full_file)

    meta_df = read_meta(base_dir, games)
    meta_df.to_csv(meta_file)


if __name__ == "__main__":
    main()
