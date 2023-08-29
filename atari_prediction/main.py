"""Module containing code for running atari prediction experiments.

Usage: python atari_main.py actions_file returns_file

Params:
        action_file - the path to the file containing the agent's actions
        returns_file - the path to the file containing the precomputed returns
"""

from tensorflow import keras
from experiment.models import HLGaussian, Regression
import sys
import json
from atari_prediction.atari_dataset import RLAdvanced
import numpy as np
from atari_prediction.base_models import value_network
from experiment.bins import get_bins
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DataCallback(keras.callbacks.Callback):
    """Callback to save model predictions at the end of each epoch.
    
    Params:
        name - the name of the model; used in the output file name
        test - the dataset containing a small number of sample batches to 
            obtain predictions for
        save_weights - flag indicating whether the model weights should be saved
            NOTE: Weight files are typically ~6 MB each
    """

    def __init__(self, name, test, save_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.test_ds = test
        self.name = name
        self.save_w = save_weights

    def on_epoch_end(self, epoch, logs=None):
        """Save model predictions and weights at the end of each epoch.
        
        Params:
            epoch - the epoch index
            logs - the model metrics for this epoch
        """
        super().on_epoch_end(epoch, logs)
        filename = f"{self.name}_{epoch}"
        if self.save_w:
            np.save(f"{filename}_w.npy", self.model.weights)        
        preds = []
        for x, y in self.test_ds:
            preds.append(self.model(x))
        np.save(f"{filename}_test.npy", np.concatenate(preds))

    
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
    learning_rate = 1e-4

    # Training params
    seed = 1
    epochs = 1
    val_ratio = 0.05
    epoch_steps = 10000
    train_steps = epoch_steps * (1 - val_ratio)
    val_steps = epoch_steps * val_ratio
    buffer_size = 1000
    batch_size = 32
    metrics = ["mse", "mae"]
    base_model = value_network
    saved_batches = 100

    # Compute the number of epoch_steps length training segments to use
    with open(action_file, "rb") as in_file:
        n = len(in_file.read())
    n_epochs = n * epochs // (epoch_steps * batch_size)

    # Get dataset and HL bins
    keras.utils.set_random_seed(seed)
    borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio)
    ds = RLAdvanced(action_file, returns_file, buffer_size=buffer_size, batch_size=batch_size)
    train, val = ds.get_split(val_ratio)
    test = ds.get_test(val_ratio)

    # Prepare callbacks for saving predictions
    val_sample = val.take(saved_batches)
    hlcb = DataCallback("HL", val_sample)
    regcb = DataCallback("Reg", val_sample)

    # Save test targets
    preds = []
    for x, y in val_sample:
        preds.append(y)
    np.save("test.npy", np.concatenate(preds))

    # Run Regression
    regression = Regression(base_model())
    regression.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=metrics)
    regression_history = regression.fit(x=train, epochs=n_epochs, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=val, callbacks=[regcb], verbose=2)
    reg_results = regression.evaluate(test, return_dict=True, verbose=2)
    with open("reg.json", "w") as file:
        json.dump(regression_history.history, file)

    # Run HL-Gaussian
    hl_gaussian = HLGaussian(base_model(), borders, sigma)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(learning_rate), metrics=metrics)
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=val, callbacks=[hlcb], verbose=2)
    hl_results = hl_gaussian.evaluate(test, return_dict=True, verbose=2)
    with open(f"hlg.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)

    results = {}
    for k, v in reg_results.items():
        results[f"reg_{k}"] = v
    for k, v in hl_results.items():
        results[f"hl_{k}"] = v
    with open("results.json", "w") as out_file:
        json.dump(results, out_file)
    


if __name__ == "__main__":
    action_file = sys.argv[1]
    returns_file = sys.argv[2]
    main(action_file, returns_file)
