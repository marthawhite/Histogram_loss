"""Module for converting the results of time series experiments to CSV."""

import json
import pandas as pd


def getjson(name, n):
    """Combine multiple JSON files into a single dict.
    
    Params:
        name - the file prefix; the JSON file has the form {name}_{i}.json
        n - the number of files
    
    Returns: a dict with the combined results
    """
    data = {}
    for i in range(n):
        file = f"{name}_{i}.json"
        with open(file, "r") as in_file:
            new_data = json.load(in_file)
        for key, val in new_data.items():
            data[key] = data.get(key, []) + val
    return data


def main():
    """Convert time series results to CSV."""
    n = 3
    outfile = "ts1.csv"
    names = ["TSHL20", "TSregression20"]
    dfs = []
    for name in names:
        data = getjson(name, n)
        df = pd.DataFrame(data)
        df["model"] = name
        dfs.append(df)
    output = pd.concat(dfs)
    output.to_csv(outfile)


if __name__ == "__main__":
    main()
