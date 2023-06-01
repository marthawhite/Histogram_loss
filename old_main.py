import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from experiment.models import HL_model, Regression, Classification
from experiment.datasets import FGNetDataset
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
    n_epochs = 50
    test_ratio = 0.2
    image_size = 128
    channels = 3
    batch_size = 32
    bins = tf.range(-10, 80, dtype=tf.float32)
    ds = FGNetDataset(data_file, size=image_size, channels=channels)
    train, test = ds.get_split(test_ratio)

    input_shape = (image_size, image_size, channels)
    
    regress = Regression(get_model(), input_shape)
    regress_history = regress.train(train, batch_size=batch_size, epochs=n_epochs, save_file="regression_model")
    with open("regression_history.json", "w") as file:
        json.dump(regress_history.history, file)
    mse = regress.validate(test)
    with open("regression_mse.txt", mode='w') as file:
        file.write(f"{mse}\n")
        
    
    hl_model = HL_model(get_model(), bins, input_shape)
    hl_history = hl_model.train(train, batch_size=batch_size, epochs=n_epochs, save_file="hl_model")
    with open("hl_history.json", "w") as file:
        json.dump(hl_history.history, file)
    mse = hl_model.validate(test)
    with open("hl_mse.txt", mode='w') as file:
        file.write(f"{mse}\n")
    
    
    classification_model = Classification(get_model(), bins, input_shape)
    classification_history = classification_model.train(train, batch_size=batch_size, epochs=n_epochs, save_file="classification_model")
    with open("classification_history.json", "w") as file:
        json.dump(classification_history.history, file)
    mse = classification_model.validate(test)
    with open("classification_mse.txt", mode='w') as file:
        file.write(f"{mse}\n")
    
    
    return

    
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)
