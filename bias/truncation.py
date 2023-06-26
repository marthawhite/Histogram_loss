import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm

def transform(inputs, borders, sigma):
    border_targets = adjust_and_erf(borders, np.expand_dims(inputs, -1), sigma)
    two_z = border_targets[:, -1] - border_targets[:, 0]
    x_trans = (border_targets[:, 1:] - border_targets[:, :-1]) / np.expand_dims(two_z, -1)
    return x_trans


def adjust_and_erf(a, mu, sig):
    return erf((a - mu) / (np.sqrt(2.0) * sig))


bin_width = 1.
y_min = 0.
y_max = 1.
sigma = 1.

pads = np.arange(1, 9, 1, np.float64)

steps = 101


# Discretization only
large_borders = np.linspace(-100, 101, 202)
large_centers = (large_borders[1:] + large_borders[:-1]) / 2
y = np.linspace(y_min, y_max, steps)
y_inf = transform(y, large_borders, sigma)
y_disc = np.dot(y_inf, large_centers)
disc_diff = y_disc - y

difs = []
maes = []
for pad in pads:

    # y_center = (y_min + y_max) / 2
    bins_min = y_min - pad * sigma
    bins_max = y_max + pad * sigma

    borders = np.arange(bins_min, bins_max + bin_width, bin_width)
    centers = (borders[1:] + borders[:-1]) / 2
    #print(borders)

    # Total error
    y_trans = transform(y, borders, sigma)
    y_new = np.dot(y_trans, centers)

    dif = y_new - y
    difs.append(dif)

    two_z = adjust_and_erf(bins_max, y, sigma) - adjust_and_erf(bins_min, y, sigma)
    bias = 2 * sigma * bin_width * (norm.pdf((bins_min - y) / sigma) - norm.pdf((bins_max - y) / sigma)) / two_z
    print(difs - disc_diff, bias)
    input()
    # if pad == 10 or pad == 11:
    #     print(dif - disc_diff)


difs = np.stack(difs)
total_maes = np.mean(np.abs(difs), axis=1)
disc_maes = np.mean(np.abs(disc_diff))
pad_maes = np.mean(np.abs(difs - disc_diff), axis=1)

#print(list(pad_maes))
#print(difs[0:3])

y, new_pads = np.meshgrid(y - 0.5, pads)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(y, new_pads, difs, cmap=cm.viridis)
ax.set_xlabel("delta / w")
ax.set_ylabel("padding / sigma")
ax.set_zlabel("Bias")
ax.set_title("Bias by padding ratio and relative bin position")
fig.colorbar(surf)
plt.show()

plt.plot(pads, total_maes, label="Total Error")
plt.plot(pads, np.resize(disc_maes, pads.shape), label="Discretization")
plt.plot(pads, pad_maes, label="Padding")
plt.xlabel("padding / sigma")
plt.ylabel("MAE")
plt.yscale("log")
plt.title("Errors by ratio of padding to sigma")
plt.legend()
plt.show()
