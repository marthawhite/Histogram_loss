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
from itertools import product
from scipy.stats import sem


def task_name_str(task_name):
    split = task_name.split('_')
    return f"Loss: {split[0]}, depth: {split[1]}, width: {split[2]}, lr: {split[3]}, Freq: {split[4]}, Offset: {split[5]}"


def main(Y_freq=4, Y_offset=0, plot_over='Y_offset', arch_over='depth'):
    
    depths = [2, 3, 4]
    widths = [1024]
    if arch_over == 'depth':
        archs = list(product(depths, [1024]))
        x = np.array(depths) + 1
    elif arch_over == 'width':
        archs = list(product([2], widths))
        x = widths

    f = h5py.File('results/sin_functions.hdf5', 'a')
    lrs_float = [2**x for x in range(-12, 5)]
    lrs = [str(lr) for lr in lrs_float]
    seeds = range(5)
    # for key in f.keys():
    #     print(key)
    # print('-')

    if plot_over == 'Y_freq':
        items = [1, 2, 4, 8]
        pallette = [hsv_to_rgb((.3, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
        bal_pallette = [hsv_to_rgb((.5, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
        l2_pallette = [hsv_to_rgb((.03, 1., ind/(len(items)+1))) for ind in range(len(items)+2)][2:]

        for ind, Y_freq in enumerate(items):
            ys = []
            stds = []
            for depth, width in archs:
                best_lr_y = np.inf
                best_lr_std = 0.
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    if np.array(errs).mean() < best_lr_y:
                        best_lr_y = np.array(errs).mean()
                        best_lr_std = sem(np.array(errs))
                ys.append(best_lr_y)
                stds.append(best_lr_std)

            ys = np.array(ys)
            stds = np.array(stds)
            plt.plot(x, ys, color=pallette[ind], label=f'Freq: {Y_freq}')
            plt.fill_between(x, ys - stds, ys + stds, color=pallette[ind], alpha=0.1)
            
        for ind, Y_freq in enumerate(items):
            ys = []
            stds = []
            for depth, width in archs:
                best_lr_y = np.inf
                best_lr_std = 0.
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss-Balanced_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    if np.array(errs).mean() < best_lr_y:
                        best_lr_y = np.array(errs).mean()
                        best_lr_std = sem(np.array(errs))
                ys.append(best_lr_y)
                stds.append(best_lr_std)

            ys = np.array(ys)
            stds = np.array(stds)
            plt.plot(x, ys, color=bal_pallette[ind], label=f'Freq: {Y_freq}')
            plt.fill_between(x, ys - stds, ys + stds, color=bal_pallette[ind], alpha=0.1)

        for ind, Y_freq in enumerate(items):
            ys = []
            stds = []
            for depth, width in archs:
                best_lr_y = np.inf
                best_lr_std = 0.
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"l2_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    if np.array(errs).mean() < best_lr_y:
                        best_lr_y = np.array(errs).mean()
                        best_lr_std = sem(np.array(errs))
                ys.append(best_lr_y)
                stds.append(best_lr_std)

            ys = np.array(ys)
            stds = np.array(stds)
            plt.plot(x, ys, color=l2_pallette[ind], label=f'Freq: {Y_freq}')
            plt.fill_between(x, ys - stds, ys + stds, color=l2_pallette[ind], alpha=0.1)

        plt.ylim(ymin=-0.01, ymax=.5)

    elif plot_over == 'Y_offset':
        items = [0, 10, 20]
        pallette = [hsv_to_rgb((.3, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
        bal_pallette = [hsv_to_rgb((.5, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
        l2_pallette = [hsv_to_rgb((.03, 1., ind/(len(items)+1))) for ind in range(len(items)+2)][2:]

        for ind, Y_offset in enumerate(items):
            ys = []
            stds = []
            for depth, width in archs:
                best_lr_y = np.inf
                best_lr_std = 0.
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_21.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    if np.array(errs).mean() < best_lr_y:
                        best_lr_y = np.array(errs).mean()
                        best_lr_std = sem(np.array(errs))
                ys.append(best_lr_y)
                stds.append(best_lr_std)

            ys = np.array(ys)
            stds = np.array(stds)
            plt.plot(x, ys, color=pallette[ind], label=f'Offset: {Y_offset}')
            plt.fill_between(x, ys - stds, ys + stds, color=pallette[ind], alpha=0.1)

        for ind, Y_offset in enumerate(items):
            ys = []
            stds = []
            for depth, width in archs:
                best_lr_y = np.inf
                best_lr_std = 0.
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss-Balanced_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_21.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    if np.array(errs).mean() < best_lr_y:
                        best_lr_y = np.array(errs).mean()
                        best_lr_std = sem(np.array(errs))
                ys.append(best_lr_y)
                stds.append(best_lr_std)

            ys = np.array(ys)
            stds = np.array(stds)
            plt.plot(x, ys, color=bal_pallette[ind], label=f'Offset: {Y_offset}')
            plt.fill_between(x, ys - stds, ys + stds, color=bal_pallette[ind], alpha=0.1)
            
        for ind, Y_offset in enumerate(items):
            ys = []
            stds = []
            for depth, width in archs:
                best_lr_y = np.inf
                best_lr_std = 0.
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"l2_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    if np.array(errs).mean() < best_lr_y:
                        best_lr_y = np.array(errs).mean()
                        best_lr_std = sem(np.array(errs))
                ys.append(best_lr_y)
                stds.append(best_lr_std)

            ys = np.array(ys)
            stds = np.array(stds)
            plt.plot(x, ys, color=l2_pallette[ind], label=f'Offset: {Y_offset}')
            plt.fill_between(x, ys - stds, ys + stds, color=l2_pallette[ind], alpha=0.1)

        plt.ylim(ymin=-0.01, ymax=.5)

    plt.ylabel('MSE')
    plt.xlabel(arch_over)
    plt.xticks(x, labels=x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{arch_over}_{plot_over}.png', dpi=200)

    f.close()


if __name__ == '__main__':
    fire.Fire(main)