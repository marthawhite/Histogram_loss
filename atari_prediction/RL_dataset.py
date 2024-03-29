"""Old Atari dataset. 

NOTE: For datasets used for recent experiments, see atari_dataset.py
"""

import tensorflow as tf
import numpy as np
import gym
import os


class Generator:
    def __init__(self, action_file, returns_file):
        game = action_file.split(os.sep)[-1].split(".")[0]
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        self.env = gym.wrappers.FrameStack(env, 4)
        self.action_file = action_file
        self.file = open(self.action_file, "rb")
        self.get_returns(returns_file)
        self.i = -1
        
        
    def __call__(self):
        while True:
            byte = self.file.read(1)
            while byte != b"":
                val = int.from_bytes(byte, 'big')
                if(val==82):
                    obs, info = self.env.reset()
                else:
                    obs, r, done, _,_ = self.env.step(val - 97)
                    self.i += 1
                    yield np.array(obs), self.outputs[self.i]
                byte = self.file.read(1)
            self.reset_file()
    
    def reset_file(self):
        self.file.seek(0)
        self.i = -1

    def get_returns(self, returns_file):
        data = np.load(returns_file)
        self.max, self.min = np.max(data), np.min(data)
        scale = self.max - self.min
        if scale == 0.:
            scale = 1.
        self.outputs = (data - self.min) / scale

    
def get_dataset(action_file, returns_file):
    gen = Generator(action_file, returns_file)
    
    inputs = tf.data.Dataset.from_generator(
        gen,
        output_signature = (tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.float32))
    )
    
    return inputs
