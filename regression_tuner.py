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
from experiment.UTKFace_loader import UTKFaceDataset


def main(base_dir):
    keras.utils.set_random_seed(1)
    n_trials = 3
    runs_per_trial = 3
    n_epochs = 40
    test_ratio = 0.1
    image_size = 128
    channels = 3
    batch_size = 32
    directory = os.path.join(base_dir, "hypers")
    
    path = os.path.join(base_dir, "data", "UTKFace")
    ds = UTKFaceDataset(path, size=image_size, channels=channels, batch_size=batch_size)
    train, test = ds.get_split(test_ratio)
    metrics = ["mse", "mae"]
    
    hp = kt.HyperParameters()
    hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])
    
    regression = HyperRegression(lambda : get_model(model="vgg16"), loss="mse")
    
    regression_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=regression, 
        objective = "val_mse", 
        directory=directory, 
        project_name=f"regression_utk", 
        overwrite=False,
        tune_new_entries=False,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial,
    ) 
    callbacks = [keras.callbacks.EarlyStopping(patience=4)]
    regression_tuner.search(x=train, epochs=n_epochs, validation_data=test, verbose=2, callbacks=callbacks)
    
    data = regression_tuner.get_results()
    model = "Regression"
    results = {}
    results[model] = data

    with open(f"regression_utk.json", "w") as out_file:
        json.dump(results, out_file, indent=4)
    
    
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
