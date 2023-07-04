import sys
sys.path.append('./')
sys.path.append('../')
import gym
from tqdm import tqdm
import numpy as np
import os
import sys
import tensorflow as tf
from experiment.datasets import Dataset
from experiment.base_model import get_base_model, test_model
from experiment.hypermodels import HyperRegression, HyperHLGaussian
import keras_tuner as kt

class RLDataset(Dataset):

    def __init__(self, policy_path, game_name, returns_path, **kwargs) -> None:
        self.policy = policy_path
        self.returns = self.get_returns(returns_path)
        self.env = self.prep_game(game_name)
        self.f = open(self.policy, "rb")
        super().__init__(**kwargs)

    def get_returns(self, path):
        data = np.load(path)
        self.bounds = np.min(data), np.max(data)
        self.scale_factor = self.bounds[1] - self.bounds[0]
        if self.scale_factor == 0.:
            self.scale_factor = 1.
        return data


    def prep_game(self, game):
        env = gym.make(game)
        env.seed(1)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    def frame_generator(self):
        byte = self.f.read(1)
        while byte != b"":
            val = int.from_bytes(byte, 'big')
            if(val==82):
                self.env.reset()
            else:
                obs, r, done, _,_ = self.env.step(val - 97)
                yield np.asarray(obs)
            byte = self.f.read(1)

    def load(self):
        ds = tf.data.Dataset.from_generator(self.frame_generator, output_signature=tf.TensorSpec(shape=(4, 84, 84), dtype=tf.uint8))
        returns_ds = tf.data.Dataset.from_tensor_slices(self.returns)
        self.ds = tf.data.Dataset.zip((ds, returns_ds))

    def preprocess(self, x, y):
        return tf.image.convert_image_dtype(x, tf.float32), self.scale(y)

    def scale(self, y):
        return (y - self.bounds[0]) / self.scale_factor

    def get_train(self):
        return self.prepare([self.ds])[0]
    

def main(base_dir):
    tf.keras.utils.set_random_seed(1)
    game = "PongNoFrameskip-v4"
    path = os.path.join(base_dir, "atari_prediction", "policies", game + ".txt")
    ds = RLDataset(path, game, "data\\returns_small.npy", buffer_size=1000)
    data = ds.get_train()
    metrics = ["mse", "mae"]
    hp = kt.HyperParameters()
    hp.Fixed("dropout", 0.)
    # reg = HyperRegression(test_model, "mse", metrics).build(hp)
    # reg.fit(data, epochs=10, steps_per_epoch=100)
    # reg.evaluate(data, steps=100)
    # reg.summary()
    reg = HyperHLGaussian(test_model, 0., 1., metrics).build(hp)
    reg.fit(data, epochs=10, steps_per_epoch=100)
    reg.evaluate(data, steps=100)
    #reg.summary()

if __name__ == "__main__":
    base_dir = sys.argv[1]
    main(base_dir)