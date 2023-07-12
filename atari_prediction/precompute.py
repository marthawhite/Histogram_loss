"""Class for precomputing the returns from Atari games and prespecified actions.
Saves the returns as {game}.npy in the returns_dir.

Usage: python precompute.py base_dir actions_file returns_dir

Params:
    base_dir - path to the directory containing the actions file
    actions_file - the name of the file containing the prespecified actions
        should be of the form {game}.txt
    returns_dir - the directory to save the returns in 
"""

import sys
sys.path.append('./')
sys.path.append('../')
import gym
import numpy as np
import os
import sys


class PolicyPrecompute:
    """Class for precomputing returns for Atari games.
    
    Params:
        policy_file - the path to the file containing the prespecified actions
        game - the name of the game to precompute returns for
        seed - the game seed; should match the seed used to generate the actions
    """

    def __init__(self, policy_file, game, seed=1) -> None:
        self.policy = policy_file
        self.game = game
        self.seed = seed
        self.returns = self.get_returns()

    def compute_return(self, cumulants, gamma, dones):
        """Calculate the returns from the rewards at each timestep.
        From sample_test.py by Esraa.

        Params:
            cumulants - the rewards obtained at each timestep
            gamma - the parameter used to exponentially weight the returns
            dones - boolean vector indicating if the game terminated in each timestep

        Returns: an exponential weighting of future rewards
        """
        num_time_steps = len(cumulants)
        returns = np.zeros(num_time_steps)
        returns[-1] = cumulants[-1]
        for t in range(num_time_steps - 2, -1, -1):
            returns[t] = gamma * returns[t + 1] * (1-dones[t]) + cumulants[t] 
        
        return returns
    
    def get_env(self):
        """Initialize the game environment.
        
        Returns: the gym environment for the game
        """
        env = gym.make(self.game)
        env.seed(self.seed)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    
    def get_returns(self):
        """Compute the returns for the game.
        
        Returns: the array of returns
        """
        env = self.get_env()

        gamma = 0.98
        rewards = []
        dones = []
        values = []
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
                if i % 1000000 == 0:
                    print(self.game, i // 1000000)

        returns = self.compute_return(rewards,gamma,dones)
        return returns
    
    def save(self, out_file):
        """Save the returns to a .npy file.
        
        Params:
            outfile - the name of the file to write the returns to
        """
        returns = self.returns.astype(np.float32)
        np.save(out_file, returns)


def main(base_dir, policy, returns_dir):
    """Compute and save the returns for a game."""
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
