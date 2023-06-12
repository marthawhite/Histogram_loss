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

def get_model():
    base_model = keras.applications.Xception(
        include_top=False,
        weights=None,
        input_tensor=layers.Input(shape=(128,128,3)),
        pooling="avg",
    )
    return base_model

def main(base_dir):
    seed = 1
    n_trials = 6
    runs_per_trial = 1
    n_epochs = 30
    image_size = 128
    channels = 3
    batch_size = 32
    y_min = 0
    y_max = 70
    directory = os.path.join(base_dir, "hypers")

    keras.utils.set_random_seed(seed)
    
    path = os.path.join(base_dir, "data", "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(path, size=image_size, channels=channels, batch_size=batch_size)
    train, test = ds.get_split(None)
    metrics = ["mse", "mae"]
    # tune regression
    hp = kt.HyperParameters()
    hp.Float("sig_ratio", min_value=0.25, max_value=8., step=2, sampling="log")
    
    hl_gaussian = HyperHLGaussian(get_model, y_min, y_max)
    
    hl_gaussian_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=hl_gaussian, 
        objective = "val_mse", 
        directory=directory, 
        project_name="hl_gaussian_sigma", 
        overwrite=False,
        tune_new_entries=False,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial
    )
    callbacks = [keras.callbacks.EarlyStopping(patience=4)]
    hl_gaussian_tuner.search(x=train, epochs=n_epochs, validation_data=test, verbose=2, callbacks=callbacks) 

    data = hl_gaussian_tuner.get_results()
    model = "HL-Gaussian"
    results = {}
    results[model] = data

    with open("hl_gaussian_results.json", "w") as out_file:
        json.dump(results, out_file, indent=4)
    
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
