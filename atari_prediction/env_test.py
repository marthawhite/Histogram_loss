"""Play an Atari game given prespecified actions and display the game window."""

import gym
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Play game
    game = "KungFuMaster"
    env_name = f"{game}NoFrameskip-v4"
    policy = os.path.join("atari_prediction", "policies", f"{env_name}.txt")
    returns_path = os.path.join("data", "returns", f"{env_name}.npy")
    returns = np.load(returns_path)
    plt.plot(returns[:1000])
    plt.show()

    env = gym.make(env_name, render_mode="human")
    env.seed(1)
    f = open(policy, "rb")
    i = -1
    for byte in f.read():
        if byte == ord('R'):
            env.reset()
        else:
            obs, r, done, _, _ = env.step(byte - ord('a'))
            i += 1
            print(i)
            if i > 1000:
                return
    env.close()


if __name__ == "__main__":
    main()
