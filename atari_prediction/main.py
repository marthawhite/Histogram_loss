"""Module containing code for running atari prediction experiments.

Usage: python atari_main.py actions_file returns_file

Params:
        action_file - the path to the file containing the agent's actions
        returns_file - the path to the file containing the precomputed returns
"""

import tensorflow as tf
from tensorflow import keras
from experiment.models import HLGaussian, Regression
import sys
import json
from atari_prediction.atari_dataset import RLAlternating
import numpy as np
from atari_prediction.base_models import value_network, large_model


class DataCallback(keras.callbacks.Callback):

    def __init__(self, name, train, test, **kwargs):
        super().__init__(**kwargs)
        self.train_ds = train
        self.test_ds = test
        self.name = name       

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        filename = f"{self.name}_{epoch}"
        preds = []
        for x, y in self.train_ds:
            preds.append(self.model(x))
        np.save(f"{filename}_train.npy", np.concatenate(preds))
        preds = []
        for x, y in self.test_ds:
            preds.append(self.model(x))
        np.save(f"{filename}_test.npy", np.concatenate(preds))


def get_bins(n_bins, pad_ratio, sig_ratio):
    """Return the histogram bins given the HL parameters.
    
    Params:
        n_bins - the number of bins to create (includes padding)
        pad_ratio - the number of sigma of padding to use on each side 
        sig_ratio - the ratio of sigma to bin width

    Returns: 
        borders - a Tensor of n_bins + 1 bin borders
        sigma - the sigma to use for HL-Gaussian
    """
    bin_width = 1 / (n_bins - 2 * sig_ratio * pad_ratio)
    pad_width = sig_ratio * pad_ratio * bin_width
    borders = tf.linspace(-pad_width, 1 + pad_width, n_bins + 1)
    sigma = bin_width * sig_ratio
    return borders, sigma

    
def main(action_file, returns_file):
    """Run the atari experiment.
    
    Params:
        action_file - the path to the file containing the agent's actions
        returns_file - the path to the file containing the precomputed returns
    """
    
    # Model params
    n_bins = 100
    pad_ratio = 4.
    sig_ratio = 2.
    learning_rate = 1e-3

    # Training params
    seed = 1
    val_ratio = 0.05
    epoch_steps = 10000
    train_steps = epoch_steps * (1 - val_ratio)
    val_steps = epoch_steps * val_ratio
    buffer_size = 1000
    batch_size = 32
    metrics = ["mse", "mae"]
    base_model = value_network
    saved_batches = 32

    with open(action_file, "rb") as in_file:
        n = len(in_file.read())
    n_epochs = n // (epoch_steps * batch_size)

    keras.utils.set_random_seed(seed)
    borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio)
    
    ds = RLAlternating(action_file, returns_file, buffer_size=buffer_size, batch_size=batch_size)
    train, val = ds.get_split(val_ratio)
    train_sample = ds.get_train(val_ratio).take(saved_batches)
    val_sample = val.take(saved_batches)
    hlcb = DataCallback("HL", train_sample, val_sample)
    regcb = DataCallback("Reg", train_sample, val_sample)

    preds = []
    for x, y in train_sample:
        preds.append(y)
    np.save("train.npy", np.concatenate(preds))

    preds = []
    for x, y in val_sample:
        preds.append(y)
    np.save("test.npy", np.concatenate(preds))

    # Run HL-Gaussian
    hl_gaussian = HLGaussian(base_model(), borders, sigma)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(learning_rate), metrics=metrics)
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=val, callbacks=[hlcb], verbose=2)
    with open(f"hlg.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)
    np.save(f"HL_w.npy", hl_gaussian.weights)

    # Run Regression
    regression = Regression(base_model())
    regression.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=metrics)
    regression_history = regression.fit(x=train, epochs=n_epochs, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=val, callbacks=[regcb], verbose=2)
    with open("reg.json", "w") as file:
        json.dump(regression_history.history, file)
    np.save(f"Reg_w.npy", regression.weights)
    

if __name__ == "__main__":
    action_file = sys.argv[1]
    returns_file = sys.argv[2]
    main(action_file, returns_file)
