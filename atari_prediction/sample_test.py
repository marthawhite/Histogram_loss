import sys
sys.path.append('./')
sys.path.append('../')
import gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

## Creating environment 
game = "Pong" +"NoFrameskip-v4"
env = gym.make(game)
# This seed is fixed and was used
# by the pre-trained agents to record actions.
# Changing it will break the benchmark
env.seed(1)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)

# Example of Value Network
class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
 
#Helper function to calculate the retuen
def compute_return(cumulants, gamma,dones):
    num_time_steps = len(cumulants)
    returns = np.zeros(num_time_steps)
    returns[-1] = cumulants[-1]
    for t in range(num_time_steps - 2, -1, -1):
        returns[t] = gamma * returns[t + 1] * (1-dones[t]) + cumulants[t] 
    
    return returns


gamma = 0.98
rewards = []
dones = []
values = []
pbar = tqdm(total = 10000000)
i = 0

with open("atari_prediction/policies/"+ game +".txt", "rb") as f:
    byte = f.read(1)
    ep_reward = 0
    while byte != b"":
        val = int.from_bytes(byte, 'big')
        if(val==82):
            print("Resetting; prev epiode return ", ep_reward)
            env.reset()
            ep_reward = 0
        else:
            obs, r, done, _,_ = env.step(val - 97)
            ep_reward += r
            rewards.append(r)
            dones.append(done)
        i +=1
        byte = f.read(1)
        pbar.update(1)

returns = compute_return(rewards,gamma,dones)

# return error
plt.plot(returns,label='returns',color='#4477AA')
plt.show()