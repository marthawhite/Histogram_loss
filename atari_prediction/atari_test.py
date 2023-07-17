import gym
from gym.utils.play import play

# env = gym.make("MontezumaRevengeNoFrameskip-v4", render_mode="human")
# key_actions = {}
# for i in range(18):
#     key_actions[(i + ord('a'),)] = i
# play(env, keys_to_action=key_actions)
# env.seed(1)
env = gym.make("VentureNoFrameskip-v4", render_mode="human")
env.seed(1)
for i in range(1):
    print(i)
    f = open("atari_prediction\\policies\\VentureNoFrameskip-v4.txt", "rb")
    for byte in f.read():
        if byte == ord('R'):
            env.reset()
        else:
            obs, r, done, _, _ = env.step(byte - 97 + i)
    env.reset(seed=1)
env.close()
