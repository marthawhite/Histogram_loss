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
from atari_prediction.atari_dataset import RLAdvanced
import numpy as np
from atari_prediction.base_models import value_network, large_model


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
    n_epochs = 30
    train_steps = 9000
    val_steps = 1000
    buffer_size = 1000
    batch_size = 32
    val_ratio = 0.1
    metrics = ["mse", "mae"]
    base_model = value_network

    keras.utils.set_random_seed(seed)
    borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio)
    
    ds = RLAdvanced(action_file, returns_file, buffer_size=buffer_size, batch_size=batch_size, prefetch=0)
    train, val = ds.get_split(val_ratio)

    # Run HL-Gaussian
    hl_gaussian = HLGaussian(base_model(), borders, sigma)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(learning_rate), metrics=metrics)
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=val, verbose=2)
    with open(f"hlg.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)
    data = hl_gaussian.predict(val.take(100))
    np.save("hlg.npy", data)

    # Run Regression
    regression = Regression(base_model())
    regression.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=metrics)
    regression_history = regression.fit(x=train, epochs=n_epochs, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=val, verbose=2)
    with open("reg.json", "w") as file:
        json.dump(regression_history.history, file)
    data = regression.predict(val.take(100))
    np.save("reg.npy", data)
    
if __name__ == "__main__":
    action_file = sys.argv[1]
    returns_file = sys.argv[2]
    main(action_file, returns_file)
