import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    games = ["AirRaid", "Asteroids", "KungFuMaster", "MsPacman", "Pong", "Tutankham"]
    for game in games:
        path = os.path.join("data", "vis_results", game)
        reg_path = os.path.join(path, "reg.npy")
        hlg_path = os.path.join(path, "hlg.npy")
        reg = np.load(reg_path)
        hlg = np.load(hlg_path)
        y_path = os.path.join(path, f"{game}NoFrameskip-v4.npy")
        y = np.load(y_path)
        max_val, min_val = np.max(y), np.min(y)
        scale = max_val - min_val
        if scale == 0.:
            scale = 1.
        y = (y - min_val) / scale

        n = reg.shape[0]
        y = y[:n]

        hlg_mae = np.mean(np.abs(y - hlg))
        hlg_rmse = np.mean(np.square(y - hlg))
        reg_mae = np.mean(np.abs(y - reg))
        reg_rmse = np.mean(np.square(y - reg))
        print(f"MAE: {hlg_mae}, MSE: {hlg_rmse}")
        print(f"MAE: {reg_mae}, MSE: {reg_rmse}")

        plt.plot(y[:1000])
        plt.plot(hlg[:1000])
        plt.plot(reg[:1000])
        plt.title(game)
        plt.show()


if __name__ == "__main__":
    main()
