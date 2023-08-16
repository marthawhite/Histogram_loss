"""Compute the simulated truncation bias over a range of padding ratios."""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from simulation import transform


def compute_difs(y, pads, sigma, y_min, y_max, bin_width):
    """Compute the bias for each combination of y and padding.
    
    Params:
        y - the input samples; shape (steps)
        pads - the padding ratios tested; shape (d)
        sigma - the value of sigma to use for the truncated Gaussian histogram transform
        y_min - the minimum value of the data range
        y_max - the maximum value of the data range
        bin_width - the desired bin width for the histogram
    
    Returns: 
        difs - the bias for each combination of y and padding; shape (d, steps)
    """
    difs = []
    for pad in pads:

        bins_min = y_min - pad * sigma
        bins_max = y_max + pad * sigma

        borders = np.arange(bins_min, bins_max + bin_width, bin_width)
        centers = (borders[1:] + borders[:-1]) / 2

        y_trans = transform(y, borders, sigma)
        y_new = np.dot(y_trans, centers)

        dif = y_new - y
        difs.append(dif)

    return np.stack(difs)


def plot_difs(y, pads, difs):
    """Plot the bias based on the padding and the bin offset.
    
    Params:
        y - the input samples; shape (steps)
        pads - the padding ratios tested; shape (d)
        difs - the bias for each combination of y and padding; shape (d, steps)
    """
    y, new_pads = np.meshgrid(y - 0.5, pads)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(y, new_pads, difs, cmap=cm.viridis)
    ax.set_xlabel(r"Offset $(\delta / w)$")
    ax.set_ylabel(r"$\psi_\sigma$")
    ax.set_zlabel("Bias")
    ax.set_title("Bias by Padding Ratio and Relative Bin Position")
    plt.show()


def plot_maes(pads, maes):
    """Plot the mean absolute bias for each padding ratio.
    
    Params:
        pads - the padding ratios tested
        maes - the mean absolute biases
    """
    plt.plot(pads, maes)
    plt.xlabel(r"$\psi_\sigma$")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title(r"Mean Absolute Bias by $\psi_\sigma$")
    plt.show()


def main():
    """Run the experiment and save the results to a file."""
    bin_width = 1.
    y_min = 0.
    y_max = 1.
    sigma = 2.
    steps = 100001

    pads = np.arange(1, 21, 0.5, np.float64)
    y = np.linspace(y_min, y_max, steps)

    difs = compute_difs(y, pads, sigma, y_min, y_max, bin_width)
    maes = np.mean(np.abs(difs), axis=1)

    plot_difs(y, pads, difs)
    plot_maes(pads, maes)

    np.save("truncation.npy", maes)
    

if __name__ == "__main__":
    main()
