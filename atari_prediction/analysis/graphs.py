"""Module for visualizing the results of an Atari experiment."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from experiment.transforms import TruncGaussHistTransform
import os
from experiment.bins import get_bins


def main():
    """Display histograms and predicted vs true values for an Atari experiment."""
    base_dir = os.path.join("data", "results")
    games = os.listdir(base_dir)
    for game in games:
        data_dir = os.path.join(base_dir, game)
        n_bins = 100
        pad_ratio = 4.
        sig_ratio = 2.
        
        hists = np.load(os.path.join(data_dir, "hists.npy"))
        y = np.load(os.path.join(data_dir, "y.npy"))
        reg = np.load(os.path.join(data_dir, "reg.npy"))

        borders, sig = get_bins(n_bins, pad_ratio, sig_ratio)
        centers = (borders[:-1] + borders[1:]) / 2.

        tght = TruncGaussHistTransform(borders, sig)
        y_trans = tght(y)

        fig, ax = plt.subplots(4, 8, figsize=(16, 10), layout="constrained", sharey=True, sharex=True)

        for i in range(hists.shape[0]):
            ax[i // 8, i % 8].plot(centers, hists[i], color='blue', alpha=1)
            ax[i // 8, i % 8].plot(centers, y_trans[i], color='orange', alpha=1)
        fig.suptitle(f"{game} Histograms")
        plt.show()

        y_pred = np.dot(hists, centers)

        plt.plot(y_pred, label="HL-Gaussian")
        plt.plot(y, label="True")
        plt.plot(reg, label="Regression")
        plt.legend()
        plt.title(f"{game} Predicted vs True Returns")
        plt.show()

        print(np.max(y), np.min(y), np.std(y), np.mean(y))


if __name__ == "__main__":
    main()
