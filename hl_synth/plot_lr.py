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
from scipy.stats import sem


def task_name_str(task_name):
    split = task_name.split('_')
    return f"Loss: {split[0]}, depth: {split[1]}, lr: {split[2]}, Freq: {split[3]}, Offset: {split[4]}"


def main(Y_freq=4, Y_offset=0, plot_over='Y_offset'):
    
    for depth in [2, 3, 4]:
        f = h5py.File('results/sin_functions.hdf5', 'a')
        lrs_float = [2**x for x in range(-12, 5)]
        lrs = [str(lr) for lr in lrs_float]
        seeds = range(5)
        # for key in f.keys():
        #     print(key)
        # print('-')

        plt.clf()
        if plot_over == 'Y_freq':
            items = [1, 2, 4, 8]
            pallette = [hsv_to_rgb((.3, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
            bal_pallette = [hsv_to_rgb((.5, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
            l2_pallette = [hsv_to_rgb((.03, 1., ind/(len(items)+1))) for ind in range(len(items)+2)][2:]
            for ind, Y_freq in enumerate(items):
                lr_curve = []
                std_curve = []
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss_{depth}_1024_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    lr_curve.append(np.array(errs).mean())
                    std_curve.append(sem(errs))
                lr_curve = np.array(lr_curve)
                std_curve = np.array(std_curve)
                plt.plot(lrs_float, lr_curve, color=pallette[ind], label=f'HL-Gauss, Freq: {Y_freq}')
                plt.fill_between(lrs_float, lr_curve - std_curve, lr_curve + std_curve, color=pallette[ind], alpha=0.2)
                lr_curve = []
                std_curve = []
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss-Balanced_{depth}_1024_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    lr_curve.append(np.array(errs).mean())
                    std_curve.append(sem(errs))
                lr_curve = np.array(lr_curve)
                std_curve = np.array(std_curve)
                plt.plot(lrs_float, lr_curve, color=bal_pallette[ind], label=f'HL-Gauss-Bal, Freq: {Y_freq}')
                plt.fill_between(lrs_float, lr_curve - std_curve, lr_curve + std_curve, color=bal_pallette[ind], alpha=0.2)
                lr_curve = []
                std_curve = []
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"l2_{depth}_1024_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    lr_curve.append(np.array(errs).mean())
                    std_curve.append(sem(errs))
                lr_curve = np.array(lr_curve)
                std_curve = np.array(std_curve)
                plt.plot(lrs_float, lr_curve, color=l2_pallette[ind], label=f'l2, Freq: {Y_freq}')
                plt.fill_between(lrs_float, lr_curve - std_curve, lr_curve + std_curve, color=l2_pallette[ind], alpha=0.2)
                plt.ylim(ymin=-0.01, ymax=.5)

        elif plot_over == 'Y_offset':
            items = [0, 10, 20]
            pallette = [hsv_to_rgb((.3, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
            bal_pallette = [hsv_to_rgb((.5, 1., ind/(len(items)+2))) for ind in range(len(items)+3)][3:]
            l2_pallette = [hsv_to_rgb((.03, 1., ind/(len(items)+1))) for ind in range(len(items)+2)][2:]
            for ind, Y_offset in enumerate(items):
                lr_curve = []
                std_curve = []
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss_{depth}_1024_{lr}_{Y_freq}_{Y_offset}_-1.5_21.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    lr_curve.append(np.array(errs).mean())
                    std_curve.append(sem(errs))
                lr_curve = np.array(lr_curve)
                std_curve = np.array(std_curve)
                plt.plot(lrs_float, lr_curve, color=pallette[ind], label=f'HL-Gauss, Offset: {Y_offset}')
                plt.fill_between(lrs_float, lr_curve - std_curve, lr_curve + std_curve, color=pallette[ind], alpha=0.2)
                lr_curve = []
                std_curve = []
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"HL-Gauss-Balanced_{depth}_1024_{lr}_{Y_freq}_{Y_offset}_-1.5_21.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    lr_curve.append(np.array(errs).mean())
                    std_curve.append(sem(errs))
                lr_curve = np.array(lr_curve)
                std_curve = np.array(std_curve)
                plt.plot(lrs_float, lr_curve, color=bal_pallette[ind], label=f'HL-Gauss-Bal, Offset: {Y_offset}')
                plt.fill_between(lrs_float, lr_curve - std_curve, lr_curve + std_curve, color=bal_pallette[ind], alpha=0.2)
                lr_curve = []
                std_curve = []
                for lr in lrs:
                    errs = []
                    for seed in seeds:
                        task_name = f"l2_{depth}_1024_{lr}_{Y_freq}_{Y_offset}_-1.5_1.5_{seed}"
                        if task_name not in f:
                            print(task_name, 'not found')
                            err = np.inf
                        else:
                            err = f[f"{task_name}/train_mse"][...].min()
                        errs.append(err)
                    lr_curve.append(np.array(errs).mean())
                    std_curve.append(sem(errs))
                lr_curve = np.array(lr_curve)
                std_curve = np.array(std_curve)
                plt.plot(lrs_float, lr_curve, color=l2_pallette[ind], label=f'l2, Offset: {Y_offset}')
                plt.fill_between(lrs_float, lr_curve - std_curve, lr_curve + std_curve, color=l2_pallette[ind], alpha=0.2)
                plt.ylim(ymin=-0.01, ymax=.5)

        plt.ylabel('MSE')
        plt.xlabel('learning rate')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/lr_{plot_over}_{depth}.png', dpi=200)

        f.close()


if __name__ == '__main__':
    fire.Fire(main)