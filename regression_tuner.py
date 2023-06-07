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


def main(base_dir, worker_num):
    n_trials = 8
    runs_per_trial = 1
    n_epochs = 30
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 32
    borders = tf.range(-10,80, 1, tf.float32)
    y_min = 0
    y_max = 70
    directory = os.path.join(base_dir, "hypers")
    
    path = os.path.join(base_dir, "data", "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(path, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size).prefetch(1)
    test = test.batch(batch_size=batch_size).prefetch(1)
    metrics = ["mse", "mae"]
    
    
    hp = kt.HyperParameters()
    hp.Choice("learning_rate", [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005])
    
    regression = HyperRegression(get_model, loss="mse")
    
    regression_tuner = LogGridSearch(
        metrics=metrics,
        hyperparameters=hp,
        hypermodel=regression, 
        objective = "val_mse", 
        directory=directory, 
        project_name="regression", 
        overwrite=False,
        tune_new_entries=False,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial,
        distribution_strategy=tf.distribute.MirroredStrategy()
        json_file = f"temp_regression_results{worker_num}.json",
    ) 
    regression_tuner.search(x=train, epochs=n_epochs, validation_data=test, verbose=2)
    
    data = regression_tuner.get_results()
    model = "Regression"
    results = {}
    results[model] = data

    with open(f"regression_results{worker_num}.json", "w") as out_file:
        json.dump(results, out_file, indent=4)
    
    
if __name__ == "__main__":
    data_file = sys.argv[1]
    worker_num = sys.argv[2]
    main(data_file, worker_num)
