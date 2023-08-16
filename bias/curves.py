"""Module for fitting curves to approximate the biases."""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


def log_f(x, a, b):
    """Log of the bias approximation function used to fit the curve."""
    return np.log(f(x, a, b))


def f(x, a, b):
    """Squared-exponential function used to model the bias."""
    return a * np.exp(b * (x ** 2))


def fit_curve(x, y, init):
    """Fit a squared-exponential function to the MAE of the bias
    using by minimizing the squares of the log ratios.
    
    Params:
        x - the input values
        y - the mean absolute bias measured for each input
        init - the initial guess for the approximation parameters; tuple (a, b)

    Returns: the fitted values for the curve after optimizing
    """
    params, _ = curve_fit(log_f, x, np.log(y), p0=init)
    a, b = params
    curve = f(x, a, b)
    mae = np.mean(np.abs(np.log(y) - np.log(curve)))
    rmse = np.sqrt(np.mean((np.log(y) - np.log(curve)) ** 2))
    print(f"Params: {params}, MAE: {mae}, RMSE: {rmse}")
    return curve


def make_plot(x, y, approx, label):
    """Draw the plot comparing bias and its approximation on a log-scale.
    
    Params:
        x - the input values
        y - the mean absolute bias measured for each input
        approx - the fitted values from the approximation function
        label - the name of the bias tested
    """
    plt.plot(x, y, label=label)
    plt.plot(x, approx, label="Approximation")
    plt.title(label)
    plt.yscale("log")
    plt.legend()
    plt.show()


def main():
    """Fit curves and show the approximation for the discretization and truncation biases."""
    disc_path = os.path.join("data", "discretization.npy")
    sigs = np.exp(np.linspace(-4, 2, 101))[:72]
    maes = np.load(disc_path)[:72]

    curve = fit_curve(sigs, maes, (0.2, -20))
    make_plot(sigs, maes, curve, "Discretization Error")

    trunc_path = os.path.join("data", "truncation.npy")
    trunc = np.load(trunc_path)[:16]
    pads = np.arange(1, 21, 0.5, np.float64)[:16]
    
    curve = fit_curve(pads, trunc, (0.25, -0.5))
    make_plot(pads, trunc, curve, "Truncation Error")


if __name__ == "__main__":
    main()
