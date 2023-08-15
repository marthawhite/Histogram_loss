import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


def f(x, a, b):
    return np.log(a * np.exp(b * (x ** 2)))


def main():
    trunc_path = os.path.join("data", "truncation.npy")
    trunc = np.load(trunc_path)[:16]
    pads = np.arange(1, 21, 0.5, np.float64)[:16]
    params, _ = curve_fit(f, pads, np.log(trunc), p0=(0.25, -0.5))
    a, b = params
    curve = a * np.exp(b * (pads ** 2))
    mae = np.mean(np.abs(np.log(trunc) - np.log(curve)))
    rmse = np.sqrt(np.mean((np.log(trunc) - np.log(curve)) ** 2))
    print(params, mae, rmse)

    plt.plot(pads, trunc, label="Truncation Error")
    plt.plot(pads, curve, label="Approximation")
    plt.yscale("log")
    plt.legend()
    plt.show()

    disc_path = os.path.join("data", "discretization.npy")
    sigs = np.exp(np.linspace(-4, 2, 101))[:72]
    maes = np.load(disc_path)[:72]
    sigs = np.asarray(sigs)

    #curve = np.exp(-20 * (sigs ** 2)) * 0.2

    params, _ = curve_fit(f, sigs, np.log(maes), p0=(0.2, -20))
    a, b = params
    curve = a * np.exp(b * (sigs ** 2))
    mae = np.mean(np.abs(np.log(maes) - np.log(curve)))
    rmse = np.sqrt(np.mean((np.log(maes) - np.log(curve)) ** 2))
    print(params, mae, rmse)


    plt.plot(sigs, maes, label="Discretization Error")
    plt.plot(sigs, curve, label="Approximation")


    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
