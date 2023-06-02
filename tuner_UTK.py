import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from experiment.hypermodels import HyperRegression, HyperHLGaussian, HyperHLOneBin
import os
import sys
import json
from experiment.UTKFace_loader import UTKFaceDataset


def get_model():
    base_model = keras.applications.Xception(
        include_top=False,
        weights=None,
        input_tensor=layers.Input(shape=(128,128,3)),
        pooling="avg",
    )
    return base_model



def main(base_dir):
    n_trials = 10
    runs_per_trial = 1
    n_epochs = 10
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 32
    borders = tf.range(-10,80, 1, tf.float32)
    y_min = 0
    y_max = 70
    directory = os.path.join(base_dir, "hypers")
    
    path = os.path.join(base_dir, "Images")
    ds = UTKFaceDataset(path, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size).prefetch(1)
    test = test.batch(batch_size=batch_size).prefetch(1)
    metrics = ["mse", "mae"]
    # tune regression
    hp = kt.HyperParameters()
    regression = HyperRegression(get_model, loss="mse", metrics=metrics)
    
    regression_tuner = kt.BayesianOptimization(
        hyperparameters=hp,
        hypermodel=regression, 
        objective = "val_mse", 
        directory=directory, 
        project_name="regression", 
        overwrite=False,
        tune_new_entries=True,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial
    ) 
    regression_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
    
    # hl gaussian
    hl_gaussian = HyperHLGaussian(get_model, y_min, y_max, metrics=metrics)
    
    hl_gaussian_tuner = kt.BayesianOptimization(
        hyperparameters=hp,
        hypermodel=hl_gaussian, 
        objective = "val_mse", 
        directory=directory, 
        project_name="hl_gaussian", 
        overwrite=False,
        tune_new_entries=True,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial
    )
    hl_gaussian_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
        
    
    # hl one bin
    hl_one_bin = HyperHLOneBin(get_model, y_min, y_max, metrics=metrics)
    
    hl_one_bin_tuner = kt.BayesianOptimization(
        hypermodel=hl_one_bin,
        hyperparameters=hp, 
        objective="val_mse", 
        directory=directory, 
        project_name="hl_one_bin",
        overwrite=False,
        tune_new_entries=True,
        max_trials=n_trials, 
        executions_per_trial=runs_per_trial
    )
    hl_one_bin_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
        
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)