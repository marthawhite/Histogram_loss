import sys
sys.path.append('./')
sys.path.append('../')
import gym
from tqdm import tqdm
import numpy as np
import os
import sys

 
#Helper function to calculate the retuen
def compute_return(cumulants, gamma,dones):
    num_time_steps = len(cumulants)
    returns = np.zeros(num_time_steps)
    returns[-1] = cumulants[-1]
    for t in range(num_time_steps - 2, -1, -1):
        returns[t] = gamma * returns[t + 1] * (1-dones[t]) + cumulants[t] 
    
    return returns

def get_returns(base_dir):
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

    gamma = 0.98
    rewards = []
    dones = []
    values = []
    pbar = tqdm(total = 10000000)
    i = 0

    path = os.path.join(base_dir, "atari_prediction", "policies", f"{game}.txt")
    with open(path, "rb") as f:
        byte = f.read(1)
        ep_reward = 0
        while byte != b"":
            val = int.from_bytes(byte, 'big')
            if(val==82):
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
    return returns

def main(base_dir):
    returns = get_returns(base_dir)
    np.savetxt("returns.txt", returns)
    np.save("returns.npy", returns)


if __name__ == "__main__":
    base_dir = sys.argv[1]
    main(base_dir)