"""Plot model progress throughout training."""

import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import sys


def main(data_path):
    # Plot the MAEs of both models at each validation segment.
    models = ["hlg", "reg"]
    for model in models:
        filename = f"{model}.json"
        path = os.path.join(data_path, filename)
        with open(path, "r") as in_file:
            data = json.load(in_file)
        df = pd.DataFrame(data)
        plt.plot(df.index + 1, df["val_mae"], label=model)
    plt.legend()
    plt.ylabel("MAE")
    plt.xlabel("Train Batches (x $10^4$)")
    plt.title("Validation MAE over Time")
    plt.show()


if __name__ == "__main__":
    data_path = sys.argv[1]
    main(data_path)
