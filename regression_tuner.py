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


def main(base_dir, index):
    n_trials = 1
    runs_per_trial = 1
    n_epochs = 40
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 32
    borders = tf.range(-10,80, 1, tf.float32)
    y_min = 0
    y_max = 70
    directory = os.path.join(base_dir, "hypers")
    
    path = os.path.join(base_dir, "data", "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(path, size=image_size, channels=channels, aligned=True)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size).prefetch(1)
    test = test.batch(batch_size=batch_size).prefetch(1)
    metrics = ["mse", "mae"]
    
    lrs = [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005]
    hp = kt.HyperParameters()
    hp.Fixed("learning_rate", lrs[index - 1])
    
    regression = HyperRegression(lambda : get_model(model="vgg16"), loss="mse")
    
    regression_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=regression, 
        objective = "val_mse", 
        directory=directory, 
        project_name=f"regression_aligned{index}", 
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

    with open(f"regression_aligned{index}.json", "w") as out_file:
        json.dump(results, out_file, indent=4)
    
    
if __name__ == "__main__":
    data_file = sys.argv[1]
    index = int(sys.argv[2])
    main(data_file, index)
