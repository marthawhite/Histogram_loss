import tensorflow as tf
import numpy as np
import gym
import os



def generator(env, actions_file):
    f = open(actions_file, "rb")
    file_length = os.stat(actions_file).st_size
    
    for i in range(file_length):
        val = int.from_bytes(byte, 'big')
        if(val==82):
            obs, info = env.reset()
            yield np.asarray(obs, dtype=np.float32)
        else:
            obs, r, done, _,_ = env.step(val - 97)
            yield np.asarray(obs, dtype=np.float32)
    
    
    
    
def get_dataset(action_file_name, returns_file):
    game = "Pong" +"NoFrameskip-v4"
    env = gym.make(game)
    #action_file_name = "PongNoFrameskip-v4.txt"
    #returns_file = "returns_file.npy"
    
    inputs = tf.data.Dataset.from_generator(
        generator,
        args = [env, action_file_name]
    )
    
    outputs = tf.data.Dataset.from_tensor(tf.convert_to_tensor(np.load(returns_file)))
    
    ds = tf.data.Dataset.zip(inouts, ouputs)
    ds = ds.shuffle(1000).batch(32).prefetch(1)
    return ds
    

