import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.special
import fire
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def task_name_str(task_name):
    split = task_name.split('_')
    return f"Loss: {split[0]}, Freq: {split[1]}, Offset: {split[2]}"


def main(Y_freq=10, Y_offset=0, plot_over='Y_freq'):
    
    f = h5py.File('results/sin_functions.hdf5', 'a')
    lrs = ['1e-05', '0.0001', '0.001', '0.01', '0.1']
    seeds = range(1)

    plt.clf()
    if plot_over == 'Y_freq':
        items = [1, 10, 20]
        pallette = [hsv_to_rgb((.3, 1., ind/(len(items)-1))) for ind in range(len(items))]
        for ind, Y_freq in enumerate(items):
            best_curve = None
            for lr in lrs:  # ['0.001']:
                curves = []
                for seed in seeds:
                    task_name = f"HL-Gauss_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve[-1] < best_curve[-1]:
                    best_curve = curve
            plt.plot(best_curve, linestyle='solid', color=pallette[ind], label=f'Freq: {Y_freq}')
            best_curve = None
            for lr in lrs:  # ['0.0001']:
                curves = []
                for seed in seeds:
                    task_name = f"l2_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve[-1] < best_curve[-1]:
                    best_curve = curve
            plt.plot(best_curve, linestyle='solid', alpha=0.5, color=pallette[ind])

    elif plot_over == 'Y_offset':
        items = [0, 1, 10]
        pallette = [hsv_to_rgb((.3, 1., ind/(len(items)-1))) for ind in range(len(items))]
        for ind, Y_offset in enumerate(items):
            best_curve = None
            for lr in lrs:  # ['0.001']:
                curves = []
                for seed in seeds:
                    task_name = f"HL-Gauss_{lr}_{Y_freq}_{Y_offset}_-1.5_11.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve[-1] < best_curve[-1]:
                    best_curve = curve
            plt.plot(best_curve, linestyle='solid', color=pallette[ind], label=f'Offset: {Y_offset}')
            best_curve = None
            for lr in lrs:  # ['0.0001']:
                curves = []
                for seed in seeds:
                    task_name = f"l2_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve[-1] < best_curve[-1]:
                    best_curve = curve
            plt.plot(best_curve, linestyle='solid', alpha=0.5, color=pallette[ind])
    
    plt.ylim(ymin=0, ymax=.6)
    # plt.xlim(xmin=0, xmax=1000)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.legend()
    plt.title('Solid: HL-Gauss, Transparent: l2')
    plt.tight_layout()
    plt.savefig(f'results/{plot_over}.png', dpi=200)

    f.close()


if __name__ == '__main__':
    fire.Fire(main)