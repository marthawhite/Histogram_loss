import tensorflow as tf
from tensorflow import keras
from keras import layers
from experiment.models import HLGaussian, Regression
import sys
import json
from experiment.atari_dataset import RLDataset
import numpy as np
from experiment.base_models import value_network, large_model


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
    keras.utils.set_random_seed(1)
    
    n_bins = 100
    pad_ratio = 4.
    sig_ratio = 2.
    dropout = 0.
    learning_rate = 1e-3

    n_epochs = 3
    buffer_size = 1000
    batch_size = 32
    val_ratio = 0.1
    metrics = ["mse", "mae"]
    base_model = value_network

    borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio)
    
    ds = RLDataset(action_file, returns_file, buffer_size=buffer_size, batch_size=batch_size)
    train, val = ds.get_split(val_ratio)

    # Run HL-Gaussian
    hl_gaussian = HLGaussian(base_model(), borders, sigma, dropout)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(learning_rate), metrics=metrics)
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, validation_data=val, verbose=2)
    with open(f"hlg.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)

    # Run Regression
    regression = Regression(base_model())
    regression.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=metrics)
    regression_history = regression.fit(x=train, epochs=n_epochs, validation_data=val, verbose=2)
    with open("reg.json", "w") as file:
        json.dump(regression_history.history, file)

    # Save samples to examine after
    for x, y in train.take(1):
        out = hl_gaussian.get_hist(x, training=False)
        np.save(f"hists.npy", out.numpy())
        np.save(f"y.npy", y.numpy())
        np.save(f"reg.npy", regression(x, training=False).numpy())
    
    
if __name__ == "__main__":
    action_file = sys.argv[1]
    returns_file = sys.argv[2]
    main(action_file, returns_file)
