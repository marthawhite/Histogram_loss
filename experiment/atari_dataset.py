import tensorflow as tf
from experiment.datasets import Dataset
import numpy as np
import os
import gym

class RLDataset(Dataset):

    def __init__(self, action_file, returns_file, game=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.returns = self.get_returns(returns_file)
        self.file = open(action_file, "rb")
        if game is None:
            game = action_file.split(os.sep)[-1].split(".")[0]
        self.env = self.get_env(game)

    def get_env(self, game):
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        return gym.wrappers.FrameStack(env, 4)

    def get_returns(self, returns_file):
        data = np.load(returns_file)
        max_val, min_val = np.max(data), np.min(data)
        scale = max_val - min_val
        if scale == 0.:
            scale = 1.
        return (data - min_val) / scale

    def train_gen(self, limit):
        n = -1
        self.reset_file()
        byte = self.file.read(1)
        while True:
            if byte == b'R':
                obs, info = self.env.reset()
                n += 1
                if n == limit:
                    return
            else:
                obs, r, done, _,_ = self.env.step(ord(byte) - 97)
                self.i += 1
                yield np.array(obs), self.returns[self.i]
            byte = self.file.read(1)

    def test_gen(self):
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
        if self.train:
            gen = self.train_gen(self.train_n)
        else:
            gen = self.test_gen()
        self.train = not self.train
        return gen

    def reset_file(self):
        self.file.seek(0)
        self.env.reset(seed=1)
        self.i = -1

    def get_split(self, val_ratio):
        self.train = True
        n = self.file.read().count(b'R')
        self.train_n = int(n * (1 - val_ratio))
        spec = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
        ds = tf.data.Dataset.from_generator(self.gen, output_signature=spec)
        ds = self.prepare([ds])[0]
        return ds, ds
