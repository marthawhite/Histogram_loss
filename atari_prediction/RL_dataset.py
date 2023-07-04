import tensorflow as tf
import numpy as np
import gym
import os



def generator(env, actions_file, file_length):
    f = open(actions_file, "rb")
    file_length = os.stat(actions_file).st_size
    for i in range(file_length):
        byte = f.read(1)
        val = int.from_bytes(byte, 'big')
        if(val==82):
            obs, info = env.reset()
            #yield np.array(obs)
        else:
            obs, r, done, _,_ = env.step(val - 97)
            yield np.array(obs)
    
    
    
    

    
    
def get_dataset():
    game = "Pong" +"NoFrameskip-v4"
    env = gym.make(game)
    action_file_name = "PongNoFrameskip-v4.txt"
    #returns_file = "returns_file.npy"
    
    inputs = tf.data.Dataset.from_generator(
        generator,
        output_signature = tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8),
        args = [env, action_file_name]
    )
    
    #outputs = tf.data.Dataset.from_tensor(tf.convert_to_tensor(np.load(returns_file)))
    
    #ds = tf.data.Dataset.zip(inouts, ouputs)
    #ds = ds.shuffle(1000).batch(32).prefetch(1)
    return inputs
    



ds = get_dataset()



    

