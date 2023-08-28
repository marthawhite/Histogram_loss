import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats


def main(dir_path):
    epochs = 35
    avgs = []
    sds = []
    preds = []
    fig, axs = plt.subplots(5, 7, figsize=(19, 9), layout="constrained")
    for mode in ["test"]:
        for i in range(epochs):
        #for i in [32]:  
            hl_path = os.path.join(dir_path, f"HL_{i}_{mode}.npy")
            reg_path = os.path.join(dir_path, f"Reg_{i}_{mode}.npy")
            hl = np.load(hl_path)
            reg = np.load(reg_path)
            y_path = os.path.join(dir_path, f"{mode}.npy")
            y = np.load(y_path)
            weights_path = os.path.join(dir_path, f"HL_{i}_w.npy")
            weights = np.load(weights_path, allow_pickle=True)
            avgs.append([np.mean(x) for x in weights[:-2]])
            sds.append([np.std(x) for x in weights[:-2]])
            print(avgs[-1])
            axs[i // 7, i % 7].plot(y[:1024])
            axs[i // 7, i % 7].plot(hl[:1024])
            axs[i // 7, i % 7].plot(reg[:1024])
    n = np.asarray([np.size(x) for x in weights[:-2]])
    plt.show()
    avgs = np.stack(avgs).T
    sds = np.stack(sds)
    ses = (sds / np.sqrt(n)).T
    alpha = 0.05
    t_stats = stats.t.ppf(1 - alpha / 2, n - 1)
    print(t_stats)
    x = range(epochs)
    for i in range(avgs.shape[0] - 2):
        plt.plot(avgs[i])
        plt.fill_between(x, avgs[i] - t_stats[i] * ses[i], avgs[i] + t_stats[i] * ses[i], alpha=0.2)
    #plt.plot(preds)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()
            

    # for i in range(epochs):
    #     hl_path = os.path.join(dir_path, f"HL_{i}_w.npy")
    #     reg_path = os.path.join(dir_path, f"Reg_{i}_w.npy")
    #     hl = np.load(hl_path, allow_pickle=True)
    #     reg = np.load(reg_path, allow_pickle=True)
    #     print(hl)
    #     print(reg)

if __name__ == "__main__":
    dir_path = sys.argv[1]
    main(dir_path)