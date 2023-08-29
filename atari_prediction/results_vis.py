import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats


def main(dir_path):
    epochs = 35
    rows, cols = 6, 6
    avgs = []
    sds = []
    fig, axs = plt.subplots(rows, cols, figsize=(19, 9), layout="constrained", sharex='all')
    for mode in ["test"]:
        for i in range(epochs):
            hl_path = os.path.join(dir_path, f"HL_{i}_{mode}.npy")
            reg_path = os.path.join(dir_path, f"Reg_{i}_{mode}.npy")
            hl = np.load(hl_path)
            reg = np.load(reg_path)
            y_path = os.path.join(dir_path, f"{mode}.npy")
            y = np.load(y_path)
            weights_path = os.path.join(dir_path, f"Reg_{i}_w.npy")
            weights = np.load(weights_path, allow_pickle=True)
            avgs.append([np.mean(x) for x in weights[:-2]])
            sds.append([np.std(x) for x in weights[:-2]])
            print(f"Epoch {i+1}: {avgs[-1]}")
            ax = axs[i // cols, i % cols]
            ax.plot(y)
            ax.plot(reg, color='tab:green')
            ax.plot(hl, color='tab:orange')

    n = np.asarray([np.size(x) for x in weights[:-2]])
    plt.show()
    avgs = np.stack(avgs)
    sds = np.stack(sds)
    ses = sds / np.sqrt(n)
    alpha = 0.05
    t_stats = stats.t.ppf(1 - alpha / 2, n - 1)
    low = (avgs - t_stats * ses).T
    high = (avgs + t_stats * ses).T
    avgs = avgs.T
    x = range(1, epochs + 1)
    for i in range(avgs.shape[0] - 2):
        plt.plot(x, avgs[i])
        plt.fill_between(x, low[i], high[i], alpha=0.2)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


if __name__ == "__main__":
    dir_path = sys.argv[1]
    main(dir_path)