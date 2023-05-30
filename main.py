import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from new_models import HLGaussian, HLOneBin, Regression
from experiment.datasets import MegaAgeDataset
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

def main(data_file):
    n_epochs = 5
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 128
    boarders - tf.linspace(0,100,100)
    
    ds = MegaAgeDataset(datafile, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size)
    test = test.batch(batch_size=batch_size)
    
    
    hl_gaussian = HLGaussian(get_model(), boarders, 1.0)
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("hl_gaussian_history.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)
    
    hl_one_bin = HLOneBin(get_model(), boarders)
    hl_one_bin_history = hl_one_bin.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("hl_one_bin_history.json", "w") as file:
        json.dump(hl_one_bin_history.history, file)
    
    regression = Regression(get_model())
    regression_history = regression.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("regression_history.json", "w") as file:
        json.dump(regression_history.history, file)
    

if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)