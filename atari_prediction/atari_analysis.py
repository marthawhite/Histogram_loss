"""Module for visualizing the results of an Atari experiment."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from experiment.models import HLGaussian
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
    with open(filename, "r") as in_file:
        data = json.load(in_file)
    return data


def get_results(data_dir):
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
    

    difs = centers - np.expand_dims(means, -1)
    sq_difs = np.square(difs)
    prods = hists * sq_difs
    totals = np.sum(prods, axis=1)
    return np.sqrt(totals)

def main():
    """Display histograms and predicted vs true values for an Atari experiment."""
    base_dir = os.path.join("data", "results")
    res = []
    for game in ["Jamesbond"]:#os.listdir(base_dir):
        
        data_dir = os.path.join(base_dir, game)
        n_bins = 100
        pad_ratio = 4.
        sig_ratio = 2.
        


        data = get_results(data_dir)
        data["game"] = game


        hists = np.load(os.path.join(data_dir, "hists.npy"))
        y = np.load(os.path.join(data_dir, "y.npy"))
        reg = np.load(os.path.join(data_dir, "reg.npy"))

        data["y_sd"] = np.std(y)
        data["reg_sd"] = np.std(reg)

        if game != "Pong":
            borders, sig = get_bins(n_bins, pad_ratio, sig_ratio)
        else:
            borders = tf.linspace(-0.25, 1.25, 100)
            sig = 0.03
        centers = (borders[:-1] + borders[1:]) / 2.

        tght = TruncGaussHistTransform(borders, sig)
        y_trans = tght(y)
        y_trans += 1e-7

        y_pred = np.dot(hists, centers)

        data["hlg_sd"] = np.std(y_pred)
        kld = entropy(y_trans, hists, axis=1)
        kld = np.where(np.isfinite(kld), kld, 5)
        data["kl_mean"] = np.mean(kld)
        data["kl_sd"] = np.std(kld)
        data["hist_sd"] = np.mean(hist_sd(hists, centers, y_pred))
        res.append(data)
    df = pd.DataFrame(res)
    df.head()
    df.to_csv("atari_james.tsv", sep="\t")


if __name__ == "__main__":
    main()
