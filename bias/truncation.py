import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from simulation import transform


def compute_difs(y, pads, sigma, y_min, y_max, bin_width):
    difs = []
    for pad in pads:

        bins_min = y_min - pad * sigma
        bins_max = y_max + pad * sigma

        borders = np.arange(bins_min, bins_max + bin_width, bin_width)
        #print(bins_min, bins_max, borders)
        centers = (borders[1:] + borders[:-1]) / 2

        y_trans = transform(y, borders, sigma)
        y_new = np.dot(y_trans, centers)

        dif = y_new - y
        difs.append(dif)

    return np.stack(difs)


def plot_difs(y, pads, difs):
    y, new_pads = np.meshgrid(y - 0.5, pads)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(y, new_pads, difs, cmap=cm.viridis)
    ax.set_xlabel(r"Offset $(\delta / w)$")
    ax.set_ylabel(r"$\psi_\sigma$")
    ax.set_zlabel("Bias")
    ax.set_title("Bias by Padding Ratio and Relative Bin Position")
    plt.show()


def plot_maes(pads, maes):
    plt.plot(pads, maes)
    plt.xlabel(r"$\psi_\sigma$")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title(r"Mean Absolute Bias by $\psi_\sigma$")
    plt.show()


def main():
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
