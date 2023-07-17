import gym
from gym.utils.play import play


env = gym.make("SkiingNoFrameskip-v4", render_mode="human")
env.seed(1)
f = open("atari_prediction\\policies\\SkiingNoFrameskip-v4.txt", "rb")
for byte in f.read():
    if byte == ord('R'):
        env.reset()
    else:
        obs, r, done, _, _ = env.step(byte - ord('a'))
env.close()
