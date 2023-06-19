import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from experiment.hypermodels import HyperRegression, HyperHLGaussian, HyperHLOneBin
import os
import sys
import json
from experiment.datasets import MegaAgeDataset
from experiment.logging import LogGridSearch
from experiment.get_model import get_model


def main(base_dir):
    keras.utils.set_random_seed(1)
    n_trials = 9
    runs_per_trial = 1
    n_epochs = 40
    test_ratio = None
    image_size = 128
    channels = 3
    batch_size = 32
    y_min = 0
    y_max = 70
    directory = os.path.join(base_dir, "hypers")

    keras.utils.set_random_seed(seed)
    
    path = os.path.join(base_dir, "data", "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(path, size=image_size, channels=channels, batch_size=batch_size, aligned=True)
    train, test = ds.get_split(None)
    metrics = ["mse", "mae"]

    # tune hypers
    hp = kt.HyperParameters()
    hp.Fixed("learning_rate", 1e-4)
    hp.Choice("n_bins", [50, 100, 200])
    hp.Choice("sig_ratio", [1., 2., 4.])

    f = lambda : get_model(model="vgg16")
    hl_gaussian = HyperHLGaussian(f, y_min, y_max)
    
    hl_gaussian_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=hl_gaussian, 
        objective = "val_mse", 
        directory=directory, 
        project_name="hlg_aligned", 
        overwrite=False,
        tune_new_entries=False,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial,
    )
    callbacks = [keras.callbacks.EarlyStopping(patience=4)]
    hl_gaussian_tuner.search(x=train, epochs=n_epochs, validation_data=test, verbose=2, callbacks=callbacks) 

    data = hl_gaussian_tuner.get_results()
    model = "HL-Gaussian"
    results = {}
    results[model] = data

    with open("hlg_aligned.json", "w") as out_file:
        json.dump(results, out_file, indent=4)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
