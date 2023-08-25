import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt


def main(dir_path):
    epochs = 35
    avgs = []
    preds = []
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
            print(avgs[-1])
            # a = [np.mean(np.abs(x)) for x in weights]
            # print(a)
            # preds.append(np.mean(reg[:300]))
            # print(preds[-1] - np.mean(y[:300]))
            # plt.plot(y)
            # plt.plot(hl)
            # plt.plot(reg)
            # plt.title(f"{mode} {i}")
            # plt.show()
    avgs = np.stack(avgs)
    for x in avgs.T:
        plt.plot(x)
    #plt.plot(preds)
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