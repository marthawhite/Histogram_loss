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
from matplotlib.legend_handler import HandlerTuple
from matplotlib import patches
import matplotlib


def task_name_str(task_name):
    split = task_name.split('_')
    return f"Loss: {split[0]}, Freq: {split[1]}, Offset: {split[2]}"


def main(Y_freq=10, Y_offset=0, plot_over='Y_freq'):
    depth=3
    width=1024
    
    f = h5py.File('results/sin_functions.hdf5', 'a')
    lrs = ['1e-05', '0.0001', '0.001', '0.01', '0.1']
    seeds = range(5)
    linewidth = 2

    plt.rcParams.update({'font.size': 22})

    plt.clf()
    plot_handles = []
    plot_labels = []
    if plot_over == 'Y_freq':
        pallette = matplotlib.colormaps["tab20c"].colors[4:7]
        l2_pallette = matplotlib.colormaps["tab20c"].colors[0:3]
        items = [1, 10, 20]
        # pallette = [hsv_to_rgb((.3, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
        # l2_pallette = [hsv_to_rgb((.03, 1., ind/(len(items)+1))) for ind in range(len(items)+2)][2:]
        for ind, Y_freq in enumerate(items):
            best_curve = None
            for lr in lrs:  # ['0.001']:
                curves = []
                for seed in seeds:
                    task_name = f"HL-Gauss_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve.mean() < best_curve.mean():
                    best_curve = curve
            plt.plot(best_curve, linestyle='solid', linewidth=linewidth, color=pallette[ind], label=f'Freq: {Y_freq}')
            best_curve = None
            for lr in lrs:  # ['0.0001']:
                curves = []
                for seed in seeds:
                    task_name = f"l2_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve.mean() < best_curve.mean():
                    best_curve = curve
            plt.plot(best_curve, linestyle='solid', linewidth=linewidth, color=l2_pallette[ind])
            hl_patch = patches.Patch(color=pallette[ind])
            l2_patch = patches.Patch(color=l2_pallette[ind])
            plot_handles.append((hl_patch, l2_patch))
            plot_labels.append(f'Freq: {Y_freq}')
    elif plot_over == 'Y_offset':
        items = [0, 1, 10]
        pallette = matplotlib.colormaps["tab20c"].colors[4:7]
        l2_pallette = matplotlib.colormaps["tab20c"].colors[0:3]
        # pallette = [hsv_to_rgb((.3, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
        # l2_pallette = [hsv_to_rgb((.03, 1., ind/(len(items)+1))) for ind in range(len(items)+2)][2:]
        for ind, Y_offset in enumerate(items):
            best_curve = None
            for lr in lrs:  # ['0.001']:
                curves = []
                for seed in seeds:
                    task_name = f"HL-Gauss_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_11.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve.mean() < best_curve.mean():
                    best_curve = curve
            hl_line,  = plt.plot(best_curve, linestyle='solid', linewidth=linewidth, color=pallette[ind], label=f'Offset: {Y_offset}')
            best_curve = None
            for lr in lrs:  # ['0.0001']:
                curves = []
                for seed in seeds:
                    task_name = f"l2_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                    curve = f[f"{task_name}/train_mse"][...]
                    curves.append(curve)
                curve = np.array(curves).mean(axis=0)
                if best_curve is None or curve.mean() < best_curve.mean():
                    best_curve = curve
            l2_line,  = plt.plot(best_curve, linestyle='solid', linewidth=linewidth, color=l2_pallette[ind])
            plt.text(-25, 0.05, "HL-Gauss.", color="tab:orange", fontsize='medium')
            plt.text(400, 0.53, "$\\ell_2$", color="tab:blue", fontsize='large')
            hl_patch = patches.Patch(color=pallette[ind])
            l2_patch = patches.Patch(color=l2_pallette[ind])
            plot_handles.append((hl_patch, l2_patch))
            plot_labels.append(f'Offset: {Y_offset}')

    plt.ylim(ymin=0, ymax=.6)
    # plt.xlim(xmin=0, xmax=1000)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.legend(handles=plot_handles, 
               labels=plot_labels,
               fontsize='x-small', 
               numpoints=1, 
               handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='upper right', 
               bbox_to_anchor=(1.1, 1.1), 
               framealpha=1
               )
    plt.gca().spines[['right', 'top']].set_visible(False)
    # plt.title('Solid: HL-Gauss, Transparent: l2')
    plt.tight_layout()
    plt.savefig(f'results/clean_{plot_over}.png', dpi=200)

    f.close()


if __name__ == '__main__':
    fire.Fire(main)