import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm

def transform(inputs, borders, sigma):
    border_targets = adjust_and_erf(borders, np.expand_dims(inputs, -1), sigma)
    two_z = border_targets[:, -1] - border_targets[:, 0]
    x_trans = (border_targets[:, 1:] - border_targets[:, :-1]) / np.expand_dims(two_z, -1)
    return x_trans


def adjust_and_erf(a, mu, sig):
    return erf((a - mu) / (np.sqrt(2.0) * sig))


bin_width = 1
y_min = 0
y_max = 1
padding_r = 10

steps = 101

borders = np.linspace(-50, 51, 102)
centers = (borders[1:] + borders[:-1]) / 2

sigs = np.exp(np.linspace(-4, 2, 101))[:72]
difs = []
maes = []
for sigma in sigs:

    # y_center = (y_min + y_max) / 2
    # bins_min = y_min - padding_r * sigma
    # bins_max = y_max + padding_r * sigma



    # #borders = np.arange(bins_min, bins_max + bin_width, bin_width)
    # centers = (borders[1:] + borders[:-1]) / 2

    # print(borders)

    y = np.linspace(y_min, y_max, steps)
    y_trans = transform(y, borders, sigma)
    y_new = np.dot(y_trans, centers)

    dif = y_new - y
    difs.append(dif)

    mae = np.mean(np.abs(dif))
    maes.append(mae)
    #print(mae)

    # if sigma > 0.5 and sigma < 1:
    #     print(sigma, dif, mae)
    #     input()



difs = np.stack(difs)
maes = np.stack(maes)
print(list(sigs))
print(list(maes))
#print(np.where(sigs < 1.4, maes, 0))

y, new_sigs = np.meshgrid(y - 0.5, np.log(sigs))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(y, new_sigs, difs, cmap=cm.viridis)
ax.set_xlabel("delta / w")
ax.set_ylabel("log(sigma / w)")
ax.set_zlabel("Bias")
ax.set_title("Bias by log(sigma) and relative bin position")
fig.colorbar(surf)
plt.show()

#print(maes)
plt.plot(sigs, maes)
plt.xlabel("sigma / w")
plt.ylabel("MAE")
plt.yscale("log")
plt.title("MAE by Ratio of sigma to bin width")
plt.show()
