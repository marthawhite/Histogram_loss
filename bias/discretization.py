import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm
from simulation import transform


def plot_difs(y, sigs, difs):
    y, new_sigs = np.meshgrid(y - 0.5, np.log(sigs))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(y, new_sigs, difs, cmap=cm.viridis)
    ax.set_xlabel(r"Offset ($\delta / w$)")
    ax.set_ylabel(r"$\log(\sigma_w)$")
    ax.set_zlabel("Bias")
    ax.set_title(r"Bias by $\log(\sigma_w)$ and Relative Bin Position")
    plt.show()


def plot_maes(sigs, maes):
    plt.plot(sigs, maes)
    plt.xlabel(r"$\sigma_w$")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title(r"Mean Absolute Bias by $\sigma_w$")
    plt.show()


def compute_difs(y, borders, sigs):
    difs = []
    centers = (borders[1:] + borders[:-1]) / 2
    for sigma in sigs:
        y_trans = transform(y, borders, sigma)
        y_new = np.dot(y_trans, centers)

        dif = y_new - y
        difs.append(dif)

    return np.stack(difs)


def main():
    bin_width = 1.
    y_min = 0
    y_max = 1.
    padding_r = 100
    steps = 100001

    y = np.linspace(y_min, y_max, steps)
    borders = np.linspace(-padding_r, 1 + padding_r, 2 + 2 * padding_r) * bin_width
    sigs = np.exp(np.linspace(-4, 2, 101))

    difs = compute_difs(y, borders, sigs)
    maes = np.mean(np.abs(difs), axis=1)

    plot_difs(y, sigs, difs)
    plot_maes(sigs, maes)

    np.save("discretization.npy", maes)
    

if __name__ == "__main__":
    main()
