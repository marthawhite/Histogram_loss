"""Module for visualizing the results of an Atari experiment."""

import numpy as np
import tensorflow as tf
from experiment.transforms import TruncGaussHistTransform
import os
import json
from scipy.stats import entropy
import pandas as pd


def get_bins(n_bins, pad_ratio, sig_ratio):
    """Return the histogram bins given the HL parameters.
    
    Params:
        n_bins - the number of bins to create (includes padding)
        pad_ratio - the number of sigma of padding to use on each side 
        sig_ratio - the ratio of sigma to bin width

    Returns: 
        borders - a Tensor of n_bins + 1 bin borders
        sigma - the sigma to use for HL-Gaussian
    """
    bin_width = 1 / (n_bins - 2 * sig_ratio * pad_ratio)
    pad_width = sig_ratio * pad_ratio * bin_width
    borders = tf.linspace(-pad_width, 1 + pad_width, n_bins + 1)
    sigma = bin_width * sig_ratio
    return borders, sigma


def get_json(filename):
    """Load the data from a JSON file.
    
    Params:
        filename - the path to the JSON file to read
    
    Returns: a dict with the JSON data
    """
    with open(filename, "r") as in_file:
        data = json.load(in_file)
    return data


def get_results(data_dir):
    """Get the results for each model.
    
    Params:
        data_dir - the directory containing the JSON files for each model

    Returns: results - a dict with the results for all metrics for each model
    """
    results = {}
    files = ["reg", "hlg"]
    for file in files:
        path = os.path.join(data_dir, f"{file}.json")
        data = get_json(path)
        for k, v in data.items():
            key = f"{file}_{k}"
            results[key] = v[-1]
    return results
        
def hist_sd(hists, centers, means):
    """Compute the standard deviation from a histogram.
    
    Params:
        hists - a tensor of histograms
        centers - the centers of the histogram bins
        means - the mean of each histogram

    Returns: the standard deviation of the probability distribution described by the histograms
    """
    difs = centers - np.expand_dims(means, -1)
    sq_difs = np.square(difs)
    prods = hists * sq_difs
    totals = np.sum(prods, axis=1)
    return np.sqrt(totals)

def main():
    """Display histograms and predicted vs true values for an Atari experiment."""
    base_dir = os.path.join("data", "results")
    out_file = "atari_meta.csv"
    res = []
    games = os.listdir(base_dir)
    for game in games:
        
        # Histogram parameters
        n_bins = 100
        pad_ratio = 4.
        sig_ratio = 2.

        # Get experiment results
        data_dir = os.path.join(base_dir, game)
        data = get_results(data_dir)
        data["game"] = game

        # Load sample predictions
        hists = np.load(os.path.join(data_dir, "hists.npy"))
        y = np.load(os.path.join(data_dir, "y.npy"))
        reg = np.load(os.path.join(data_dir, "reg.npy"))

        # Compute standard deviations
        data["y_sd"] = np.std(y)
        data["reg_sd"] = np.std(reg)

        # Compute histogram borders
        if game != "Pong":
            borders, sig = get_bins(n_bins, pad_ratio, sig_ratio)
        else:
            borders = tf.linspace(-0.25, 1.25, 100)
            sig = 0.03
        centers = (borders[:-1] + borders[1:]) / 2.

        # Get transformed Y
        tght = TruncGaussHistTransform(borders, sig)
        y_trans = tght(y)
        y_trans += 1e-7
        
        y_pred = np.dot(hists, centers)

        # Compute histogram metrics
        data["hlg_sd"] = np.std(y_pred)
        kld = entropy(y_trans, hists, axis=1)
        kld = np.where(np.isfinite(kld), kld, 5)
        data["kl_mean"] = np.mean(kld)
        data["kl_sd"] = np.std(kld)
        data["hist_sd"] = np.mean(hist_sd(hists, centers, y_pred))
        res.append(data)
    
    # Save results
    df = pd.DataFrame(res)
    df.head()
    df.to_csv(out_file)


if __name__ == "__main__":
    main()
