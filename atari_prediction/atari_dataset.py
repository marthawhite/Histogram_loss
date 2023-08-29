"""Class for atari prediction RL datasets.

The current one is RLAlternating which uses two Gym instances with the test set
consisting of every k-th episode.
"""

import tensorflow as tf
from experiment.dataset import Dataset
import numpy as np
import os
import gym


class RLDataset(Dataset):
    """A dataset containing observations and returns for an RL agent from an atari game.
    
    Params:
        action_file - path to file containing the agent's actions
        returns_file - path to file containing the precomputed returns
        game - the name of the game
        kwargs - dataset superclass arguments (batch_size, buffer_size, prefetch)

    NOTE: This dataset returns a function that alternates between returning training and testing generators.
        For correct results, always alternate between training and testing iterations.
    """

    def __init__(self, action_file, returns_file, game=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.returns = self.get_returns(returns_file)
        self.file = open(action_file, "rb")
        if game is None:
            game = action_file.split(os.sep)[-1].split(".")[0]
        self.env = self.get_env(game)

    def get_env(self, game):
        """Initialize the game environment.
        
        Params:
            game - the name of the game environment

        Returns: the gym environment
        """
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        return gym.wrappers.FrameStack(env, 4)

    def get_returns(self, returns_file):
        """Load and scale the returns from a file.
        
        Params:
            returns_file - the path to the .npy file containing the returns

        Returns: the returns scaled to [0, 1]
        """
        data = np.load(returns_file)
        max_val, min_val = np.max(data), np.min(data)
        scale = max_val - min_val
        if scale == 0.:
            scale = 1.
        return (data - min_val) / scale

    def train_gen(self, limit):
        """Generate training samples for a number of runs.
        
        Params:
            limit - the number of game iterations (runs) used for training

        Yields: (obs, return)
            obs - the (4, 84, 84) image stack as a numpy array
            return - the scaled return for the corresponding timestep
        """
        n = -1
        self.reset_file()
        byte = self.file.read(1)
        while True:
            if byte == b'R':
                # Run finished
                obs, info = self.env.reset()
                n += 1
                if n == limit:
                    return
            else:
                # Yield observation
                obs, r, done, _,_ = self.env.step(ord(byte) - 97)
                self.i += 1
                yield np.array(obs), self.returns[self.i]
            byte = self.file.read(1)

    def test_gen(self):
        """Generate test samples until the end of the file.
        
        Yields: (obs, return)
            obs - the (4, 84, 84) image stack as a numpy array
            return - the scaled return for the corresponding timestep
        """
        byte = self.file.read(1)
        while byte != b"":
            if byte == b'R':
                obs, info = self.env.reset()
            else:
                obs, r, done, _,_ = self.env.step(ord(byte) - 97)
                self.i += 1
                yield np.array(obs), self.returns[self.i]
            byte = self.file.read(1)
        return
    

    def gen(self):
        """Return a generator to generate train or test samples.
        NOTE: Alternates between producing train and test generators each time it is called.

        Returns: a generator yielding (obs, return) pairs where obs is a (4, 84, 84) numpy image stack
        """
        if self.train:
            gen = self.train_gen(self.train_n)
        else:
            gen = self.test_gen()
        self.train = not self.train
        return gen

    def reset_file(self):
        """Reset the actions file at the beginning of an epoch."""
        self.file.seek(0)
        self.env.reset(seed=1)
        self.i = -1

    def get_test(self):
        """Return a test dataset without shuffling."""
        spec = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
        ds = tf.data.Dataset.from_generator(self.test_gen, output_signature=spec)
        ds = ds.batch(self.batch_size).prefetch(self.prefetch)
        return ds

    def get_split(self, val_ratio):
        """Return a dataset that allows train/test split iteration.
        
        Params:
            val_ratio - the proportion of samples to use in the test split
        
        Returns: (ds, ds)
            ds - a generator dataset that alternates between producing train and test samples each time iteration starts
                NOTE: Always alternate between training and validation when using this dataset!
        """
        self.train = True
        n = self.file.read().count(b'R')
        self.train_n = int(n * (1 - val_ratio))
        spec = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
        ds = tf.data.Dataset.from_generator(self.gen, output_signature=spec)
        ds = self.prepare([ds])[0]
        return ds, ds
    

class RLAdvanced(Dataset):
    """A dataset using two Gym instances with the test set at the beginning.
    This allows validation in the middle of training to gauge model progress.
    
    Params:
        action_file - path to file containing the agent's actions
        returns_file - path to file containing the precomputed returns
        game - the name of the game
        kwargs - dataset superclass arguments (batch_size, buffer_size, prefetch)
    """

    def __init__(self, action_file, returns_file, game=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.returns = self.get_returns(returns_file)
        self.file = action_file
        if game is None:
            self.game = action_file.split(os.sep)[-1].split(".")[0]
        else:
            self.game = game

    def get_env(self, game):
        """Initialize the game environment.
        
        Params:
            game - the name of the game environment

        Returns: the gym environment
        """
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        return gym.wrappers.FrameStack(env, 4)

    def get_returns(self, returns_file):
        """Load and scale the returns from a file.
        
        Params:
            returns_file - the path to the .npy file containing the returns

        Returns: the returns scaled to [0, 1]
        """
        data = np.load(returns_file)
        max_val, min_val = np.max(data), np.min(data)
        scale = max_val - min_val
        if scale == 0.:
            scale = 1.
        return (data - min_val) / scale

    def test_gen(self, limit):
        """Generate testing samples for a number of runs.
        
        Params:
            limit - the number of game iterations (runs) used for testing

        Yields: (obs, return)
            obs - the (4, 84, 84) image stack as a numpy array
            return - the scaled return for the corresponding timestep
        """
        n = -1
        i = -1
        env = self.get_env(self.game)
        file = open(self.file, "rb")
        byte = file.read(1)
        while n < limit:
            if byte == b'R':
                # Run finished
                obs, info = env.reset()
                n += 1
            else:
                # Yield observation
                obs, r, done, _,_ = env.step(ord(byte) - 97)
                i += 1
                yield np.array(obs), self.returns[i]
            byte = file.read(1)
        file.close()
        return


    def train_gen(self, start):
        """Generate train samples until the end of the file.
        
        Params:
            start - the index of the first episode to use for training

        Yields: (obs, return)
            obs - the (4, 84, 84) image stack as a numpy array
            return - the scaled return for the corresponding timestep
        """
        n = -1
        i = -1
        env = self.get_env(self.game)
        file = open(self.file, "rb")
        byte = file.read(1)
        while byte != b"":
            if byte == b'R':
                obs, info = env.reset()
                n += 1
            else:
                obs, r, done, _,_ = env.step(ord(byte) - 97)
                i += 1
                if n >= start:
                    yield np.array(obs), self.returns[i]
            byte = file.read(1)
        file.close()
        return
    

    def reset_file(self):
        """Reset the actions file at the beginning of an epoch."""
        self.file.seek(0)
        self.env.reset(seed=1)
        self.i = -1

    def get_split(self, val_ratio):
        """Return a dataset that allows train/test split iteration.
        
        Params:
            val_ratio - the proportion of episodes to use in the test split
        
        Returns: a tuple (train, test)
            train - the shuffled and batched train dataset
            test - the unshuffled test dataset from the beginning of the actions
        """
        with open(self.file, "rb") as in_file:
            n = in_file.read().count(b'R')
        test_n = n - int(n * (1 - val_ratio))
        spec = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
        train = tf.data.Dataset.from_generator(lambda : self.train_gen(test_n), output_signature=spec)
        train = train.repeat().shuffle(self.buf).batch(self.batch_size).prefetch(self.prefetch)
        test = tf.data.Dataset.from_generator(lambda : self.test_gen(test_n), output_signature=spec)
        test = test.repeat().batch(self.batch_size).prefetch(self.prefetch)
        return train, test


class RLAlternating(Dataset):
    """A dataset where test samples are drawn from throughout the actions file.
    Every k-th episode is used as test data where k is approximately 1 / val_ratio.
    
    Params:
        action_file - path to file containing the agent's actions
        returns_file - path to file containing the precomputed returns
        game - the name of the game
        kwargs - dataset superclass arguments (batch_size, buffer_size, prefetch)
    """

    def __init__(self, action_file, returns_file, game=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.returns = self.get_returns(returns_file)
        self.file = action_file
        if game is None:
            self.game = action_file.split(os.sep)[-1].split(".")[0]
        else:
            self.game = game

    def get_env(self, game):
        """Initialize the game environment.
        
        Params:
            game - the name of the game environment

        Returns: the gym environment
        """
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        return gym.wrappers.FrameStack(env, 4)

    def get_returns(self, returns_file):
        """Load and scale the returns from a file.
        
        Params:
            returns_file - the path to the .npy file containing the returns

        Returns: the returns scaled to [0, 1]
        """
        data = np.load(returns_file)
        max_val, min_val = np.max(data), np.min(data)
        scale = max_val - min_val
        if scale == 0.:
            scale = 1.
        return (data - min_val) / scale

    def test_gen(self, cycle):
        """Generate test samples for a number of runs.
        
        Params:
            cycle - the number of episodes between each test episode

        Yields: (obs, return)
            obs - the (4, 84, 84) image stack as a numpy array
            return - the scaled return for the corresponding timestep
        """
        n = -1
        i = -1
        env = self.get_env(self.game)
        with open(self.file, "rb") as file:
            actions = file.read()
        for byte in actions:
            if byte == 82:
                # Run finished
                obs, info = env.reset()
                n += 1
            else:
                # Yield observation
                obs, r, done, _,_ = env.step(byte - 97)
                i += 1
                if n % cycle == 0:
                    yield np.array(obs), self.returns[i]
        return


    def train_gen(self, cycle):
        """Generate train samples until the end of the file.
        
        Params:
            cycle - the length of the cycle between each test episode

        Yields: (obs, return)
            obs - the (4, 84, 84) image stack as a numpy array
            return - the scaled return for the corresponding timestep
        """
        n = -1
        i = -1
        env = self.get_env(self.game)
        with open(self.file, "rb") as file:
            actions = file.read()
        for byte in actions:
            if byte == 82:
                obs, info = env.reset()
                n += 1
            else:
                obs, r, done, _,_ = env.step(byte - 97)
                i += 1
                if n % cycle != 0:
                    yield np.array(obs), self.returns[i]
        return
    

    def reset_file(self):
        """Reset the actions file at the beginning of an epoch."""
        self.file.seek(0)
        self.env.reset(seed=1)
        self.i = -1

    def get_split(self, val_ratio):
        """Return a dataset that allows train/test split iteration.
        
        Params:
            val_ratio - the proportion of episodes to use in the test split
        
        Returns: a tuple (train, test)
            train - the shuffled and batched train dataset
            test - the unshuffled test dataset spread throughout the actions
        """
        cycle = int(1 / val_ratio)
        spec = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
        train = tf.data.Dataset.from_generator(lambda : self.train_gen(cycle), output_signature=spec)
        train = train.repeat().shuffle(self.buf).batch(self.batch_size).prefetch(self.prefetch)
        test = tf.data.Dataset.from_generator(lambda : self.test_gen(cycle), output_signature=spec)
        test = test.repeat().batch(self.batch_size).prefetch(self.prefetch)
        return train, test
    
    def get_train(self, val_ratio):
        """Return an unshuffled train dataset used for prediction visualization.
        
        Params:
            val_ratio - the proportion of episodes used in the test split

        Returns: 
            train - the unshuffled training dataset 
        """
        cycle = int(1 / val_ratio)
        spec = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
        train = tf.data.Dataset.from_generator(lambda : self.train_gen(cycle), output_signature=spec)
        train = train.repeat().batch(self.batch_size).prefetch(self.prefetch)
        return train