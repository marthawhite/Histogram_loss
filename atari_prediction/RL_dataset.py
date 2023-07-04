import tensorflow as tf
import numpy as np
import gym
import os



class Generator:
    def __init__(self, action_file):
        game = "Pong" +"NoFrameskip-v4"
        self.env = gym.make(game)
        self.action_file = action_file
        
    def __call__(self):
        f = open(self.action_file)
        file_length = os.stat(self.actions_file).st_size
        for i in range(file_length):
            byte = f.read(1)
            val = int.from_bytes(byte, 'big')
            if(val==82):
                obs, info = self.env.reset()
                yield np.array(obs)
            else:
                obs, r, done, _,_ = self.env.step(val - 97)
                yield np.array(obs)

    
    

    
    
def get_dataset(returns_file):
    gen = Generator(action_file = "PongNoFrameskip-v4.txt")
    
    inputs = tf.data.Dataset.from_generator(
        gen,
        output_signature = tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8)
    )
    
    outputs = tf.data.Dataset.from_tensor(tf.convert_to_tensor(np.load(returns_file)))
    
    ds = tf.data.Dataset.zip(inouts, ouputs)
    ds = ds.shuffle(1000).batch(32).prefetch(1)
    return ds
    





    

