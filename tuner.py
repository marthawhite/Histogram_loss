import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from hypermodels import HyperRegression, HyperHLGaussian, HyperHLOneBin
import os
import sys
import json
from experiment.datasets import MegaAgeDataset


def get_model():
    base_model = keras.applications.Xception(
        include_top=False,
        weights=None,
        input_tensor=layers.Input(shape=(128,128,3)),
        pooling="avg",
    )
    return base_model



def main(datafile):
    n_epochs = 8
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 32
    borders = tf.range(-10,80, 1, tf.float32)
    y_min = 0
    y_max = 70
    directory = "hypermodels"
    
    path = os.path.join(data_file, "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(datafile, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size)
    test = test.batch(batch_size=batch_size)
    
    # tune regression
    hp = kt.HyperParameters()
    regression = HyperRegression(get_model)
    
    regression_tuner = kt.BayesianOptimization(
        hyperparameters=hp,
        hypermodel=regression, 
        objective = "val_mse", 
        directory=directory, 
        project_name="regression", 
        tune_new_entries=True) 
    regression_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
    
    # hl gaussian
    hl_gaussian = HyperHLGaussian(get_model)
    
    hl_gaussian_tuner = kt.BayesianOptimization(hl_gaussian, objective="val_mse", overwrite=True, directory=directory, project_name="hl_gaussian")
    hl_gaussian_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
        
    
    # hl one bin
    hl_one_bin = HyperHLOneBin(get_model)
    
    hl_one_bin_tuner = kt.BayesianOptimization(hl_one_bin, objective="val_mse", directory=directory, project_name="hl_one_bin")
    hl_one_bin_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
        
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)