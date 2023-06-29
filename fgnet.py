import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from experiment.hypermodels import HyperRegression, HyperHLGaussian, HyperHLOneBin
import os
import sys
import json
from experiment.datasets import FGNetDataset
from experiment.logging import LogGridSearch
from experiment.get_model import get_model
from experiment.preprocessing import Scaler


def main(base_dir):
    keras.utils.set_random_seed(1)
    n_trials = 10
    runs_per_trial = 1
    n_epochs = 40
    test_ratio = 0.1
    image_size = 128
    channels = 3
    batch_size = 32
    y_min = 0.
    y_max = 70.
    directory = os.path.join(base_dir, "hypers")
    
    path = os.path.join(base_dir, "data", "FGNET", "aligned")
    ds = FGNetDataset(path, size=image_size, channels=channels, batch_size=batch_size)
    train, test = ds.get_split(test_ratio, shuffle=True)
    sc = Scaler(y_min, y_max)
    train = sc.transform(train)
    test = sc.transform(test)
    metrics = ["mse", "mae"]

    # tune hypers
    hp = kt.HyperParameters()
    hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    hp.Fixed("n_bins", 100)
    hp.Fixed("sig_ratio", 2.)

    results = {}

    f = lambda : get_model(model="vgg16")
    hl_gaussian = HyperHLGaussian(f, y_min, y_max)
    
    hl_gaussian_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=hl_gaussian, 
        objective = "val_mse", 
        directory=directory, 
        project_name="fgnet_hlg", 
        overwrite=False,
        tune_new_entries=False,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial,
    )
    callbacks = [keras.callbacks.EarlyStopping(patience=10)]
    hl_gaussian_tuner.search(x=train, epochs=n_epochs, validation_data=test, verbose=2, callbacks=callbacks)
    results["HL-Gaussian"] = hl_gaussian_tuner.get_results()

    l2 = HyperRegression(f, loss="mse")
    
    l2_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=l2, 
        objective = "val_mse", 
        directory=directory, 
        project_name="fgnet_reg", 
        overwrite=False,
        tune_new_entries=False,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial,
    )
    l2_tuner.search(x=train, epochs=n_epochs, validation_data=test, verbose=2, callbacks=callbacks) 

    results["Regression"] = l2_tuner.get_results()

    with open("fgnet.json", "w") as out_file:
        json.dump(results, out_file, indent=4)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
