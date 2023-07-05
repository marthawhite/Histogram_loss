import tensorflow as tf
import numpy as np
import gym
import os
import cv2


class Generator:
    def __init__(self, action_file):
        game = action_file.split(os.sep)[-1].split(".")[0]
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        self.env = gym.wrappers.FrameStack(env, 4)
        self.action_file = action_file
        self.file = open(self.action_file, "rb")
        
        
    def __call__(self):
        byte = self.file.read(1)
        while byte != b"":
            val = int.from_bytes(byte, 'big')
            if(val==82):
                obs, info = self.env.reset()
            else:
                obs, r, done, _,_ = self.env.step(val - 97)
                yield np.array(obs)
            byte = self.file.read(1)
        self.reset_file()

    
    def reset_file(self):
        self.file.seek(0)
    
    
def get_dataset(action_file, returns_file):
    gen = Generator(action_file)
    
    inputs = tf.data.Dataset.from_generator(
        gen,
        output_signature = tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8)
    )
    
    outputs = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.load(returns_file)))
    
    ds = tf.data.Dataset.zip((inputs, outputs))
    return ds
