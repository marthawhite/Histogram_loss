import sys
sys.path.append('./')
sys.path.append('../')
import gym
from tqdm import tqdm
import numpy as np
import os
import sys


class PolicyPrecompute:

    def __init__(self, policy_file, game, seed=1) -> None:
        self.policy = policy_file
        self.game = game
        self.seed = seed
        self.returns = self.get_returns()


    #Helper function to calculate the retuen
    def compute_return(self, cumulants, gamma,dones):
        num_time_steps = len(cumulants)
        returns = np.zeros(num_time_steps)
        returns[-1] = cumulants[-1]
        for t in range(num_time_steps - 2, -1, -1):
            returns[t] = gamma * returns[t + 1] * (1-dones[t]) + cumulants[t] 
        
        return returns

    def get_returns(self):
        ## Creating environment 
        env = gym.make(self.game)
        # This seed is fixed and was used
        # by the pre-trained agents to record actions.
        # Changing it will break the benchmark
        env.seed(self.seed)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        gamma = 0.98
        rewards = []
        dones = []
        values = []
        #pbar = tqdm(total = 10000000)
        i = 0

        with open(self.policy, "rb") as f:
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
                #pbar.update(1)
                if i % 100000 == 0:
                    print(self.game, i // 100000)
                    break

        returns = self.compute_return(rewards,gamma,dones)
        return returns
    
    def save(self, out_file):
        returns = self.returns.astype(np.float32)
        np.save(out_file, returns)


def main(base_dir, policy, returns_dir):
    game = policy.split(".")[0]
    policy_path = os.path.join(base_dir, policy)
    precmp = PolicyPrecompute(policy_path, game)
    returns_path = os.path.join(returns_dir, game)
    precmp.save(returns_path)


if __name__ == "__main__":
    base_dir = sys.argv[1]
    policy = sys.argv[2]
    returns_dir = sys.argv[3]
    main(base_dir, policy, returns_dir)