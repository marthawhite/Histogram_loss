import gym
import os


def main():
    # Play game
    game = "Skiing"
    env_name = f"{game}NoFrameskip-v4"
    policy = os.path.join("atari_prediction", "policies", f"{env_name}.txt")

    env = gym.make(env_name, render_mode="human")
    env.seed(1)
    f = open(policy, "rb")
    for byte in f.read():
        if byte == ord('R'):
            env.reset()
        else:
            obs, r, done, _, _ = env.step(byte - ord('a'))
    env.close()


if __name__ == "__main__":
    main()
