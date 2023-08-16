"""Module for approximating the histogram transformation bias using simulated data."""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm


def transform(inputs, borders, sigma):
    """Transform the inputs to binned probability vectors 
    using a truncated Gaussian distribution.
    
    Params:
        inputs - array of input samples; shape (steps)
        borders - array of histogram bin borders; shape (n_bins + 1)
        sigma - scale parameter for truncated Gaussian distribution
    
    Returns: the transformed inputs; shape (steps, n_bins)
    """
    border_targets = adjust_and_erf(borders, np.expand_dims(inputs, -1), sigma)
    two_z = border_targets[:, -1] - border_targets[:, 0]
    x_trans = (border_targets[:, 1:] - border_targets[:, :-1]) / np.expand_dims(two_z, -1)
    return x_trans


def adjust_and_erf(a, mu, sig):
    """Compute the erf after standardizing and dividing by sqrt(2)."""
    return erf((a - mu) / (np.sqrt(2.0) * sig))


def main():
    """Run the simulation and print the borders and MAE."""
    n_bins = 100
    bin_width = 1.
    y_min = 0.
    y_max = 1e6
    sig = 1.4
    padding = 6.5

    steps = int(1e6) + 1

    bin_size = np.abs((y_max - y_min) / (n_bins - 2 * padding * sig))
    pad_min = y_min - padding * bin_size * sig
    pad_max = y_max + padding * bin_size * sig

    y = np.linspace(y_min, y_max, steps)
    borders = np.linspace(pad_min, pad_max, n_bins + 1)
    centers = (borders[1:] + borders[:-1]) / 2

    print(bin_size, borders)

    y_trans = transform(y, borders, sig * bin_size)
    y_new = np.dot(y_trans, centers)

    dif = y_new - y
    mae = np.mean(np.abs(dif))
    print(mae)


if __name__ == "__main__":
    main()
