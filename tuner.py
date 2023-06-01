import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from hypermodels import HyperRegression, HyperHLGaussian, HyperHLOneBin
import os
import sys
import json


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
    
    path = os.path.join(data_file, "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(datafile, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size)
    test = test.batch(batch_size=batch_size)
    
    # tune regression
    #hp = kt.HyperParameters()
    regression = HyperRegression(get_model())
    
    regression_tuner = kt.BayesianOptimization(regression, objective = "val_loss") # objective = "val_accuracy" or "val_loss"
    regression_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
    best_hypers_regression = regression_tuner.get_best_hyperparameters(1)[0].values
    with open("best_regression_hypers.json", "w") as file:
        json.dump(best_hypers_regression, file)
        
    # save the best model
    best_regression_model = regression_tuner.get_best_models(1)[0]
    best_regression_model.save("regression_model")
    
    
    # hl gaussian
    hl_gaussian = HyperHLGaussian(get_model())
    
    hl_gaussian_tuner = kt.BayesianOptimization(hl_gaussian, objective="val_loss")
    hl_gaussian_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
    best_hypers_hl_gaussian = hl_gaussian_tuner.get_best_hyperparameters(1)[0].values
    with open("best_hl_gaussian_hypers.json", "w") as file:
        json.dump(best_hypers_hl_gaussian)
        
    # save the best model
    best_hl_gaussian_model = hl_gaussian_tuner.get_best_models(1)[0]
    best_hl_gaussian_model.save("hl_gaussian_model")
    
    # hl one bin
    hl_one_bin = HyperHLOneBin(get_model())
    
    hl_one_bin_tuner = kt.BayesianOptimization(hl_one_bin, objective="val_loss")
    hl_one_bin_tuner.search(x=train, epochs=n_epochs, validation_data=test)
    
    best_hypers_hl_one_bin = hl_one_bin_tuner.get_best_hyperparameters(1)[0].values
    with open("best_hl_one_bin_hypers.json", "w") as file:
        json.dump(best_hypers_hl_one_bin)
        
    # save the best model
    best_hl_one_bin_model = hl_one_bin_tuner.get_best_models(1)[0]
    best_hl_one_bin_model.save("hl_one_bin_model")
        
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)