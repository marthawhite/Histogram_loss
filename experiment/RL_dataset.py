import tensorflow as tf
import numpy as np
import gym
import os
import cv2



class Generator:
    def __init__(self, action_file):
        game = "Pong" +"NoFrameskip-v4"
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        sefl.env = gym.wrappers.FrameStack(env, 4)
        self.action_file = action_file
        self.file = open(self.action_file)
        
        
    def __call__(self):
        byte = self.file.read(1)
        while byte != b"":
            val = int.from_bytes(byte, 'big')
            if(val==82):
                obs, info = self.env.reset()
                yield np.array(obs)
            else:
                obs, r, done, _,_ = self.env.step(val - 97)
                yield np.array(obs)
            byte = self.file.read(1)
        self.reset_file()

    
    def reset_file(self):
        self.file.seek(0)

    
    
def get_dataset(returns_file):
    gen = Generator(action_file = "PongNoFrameskip-v4.txt")
    
    inputs = tf.data.Dataset.from_generator(
        gen,
        output_signature = tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8)
    )
    
    outputs = tf.data.Dataset.from_tensor(tf.convert_to_tensor(np.load(returns_file)))
    
    ds = tf.data.Dataset.zip(inputs, ouputs)
    return ds
    




    

