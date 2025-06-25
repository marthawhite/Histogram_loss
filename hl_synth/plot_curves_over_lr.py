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


def main(model_name='l2', Y_freq=1, Y_offset=1, hl_range=[-1.5, 1.5]):
    
    f = h5py.File('results/sin_functions.hdf5', 'a')
    lrs = [2**x for x in range(-12, 1)]
    found_lrs = []
    for lr in lrs:
        task_name = f'{model_name}_2_1024_{lr}_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}_{0}'
        if task_name in f:
            found_lrs.append(lr)

    plt.clf()
    pallette = [hsv_to_rgb((.3, 1., ind/(len(found_lrs)-1))) for ind in range(len(found_lrs))]
    for ind, lr in enumerate(found_lrs):
        task_name = f'{model_name}_2_1024_{lr}_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}_{0}'
        curve = f[f"{task_name}/train_mse"][...]#[:50]
        plt.plot(curve, linestyle='solid', color=pallette[ind], label=f'lr: {lr}')
    
    plt.ylim(ymin=0, ymax=1.)
    # plt.xlim(xmin=0, xmax=1000)
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.legend()
    # plt.title('Solid: HL-Gauss, Transparent: l2')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_curves_over_lr_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}.png', dpi=200)

    f.close()


if __name__ == '__main__':
    fire.Fire(main)