import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from experiment.new_models import HLGaussian, HLOneBin, Regression
from experiment.datasets import MegaAgeDataset
import os
import sys
import json
import cv2


def get_model():
    base_model = keras.applications.Xception(
        include_top=False,
        weights=None,
        input_tensor=layers.Input(shape=(128,128,3)),
        pooling="avg",
    )
    return base_model

def main(data_file):
    n_epochs = 8
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 32
    borders = tf.range(-10,80, 1, tf.float32)
    
    path = os.path.join(data_file, "megaage_asian", "megaage_asian")
    ds = MegaAgeDataset(path, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)
    train = train.batch(batch_size=batch_size).prefetch(1)
    test = test.batch(batch_size=batch_size).prefetch(1)

    hl_gaussian = HLGaussian(get_model(), borders, 1.0)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("hl_gaussian_history.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)
    
    hl_one_bin = HLOneBin(get_model(), borders)
    hl_one_bin.compile(optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    hl_one_bin_history = hl_one_bin.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("hl_one_bin_history.json", "w") as file:
        json.dump(hl_one_bin_history.history, file)
    
    regression = Regression(get_model())
    regression.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    regression_history = regression.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("regression_history.json", "w") as file:
        json.dump(regression_history.history, file)
    

if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)