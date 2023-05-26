import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models import HL_model, Regression, Classification
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
    bins = tf.range(-10, 80, dtype=tf.float32)
    #path = os.path.join("data", "FGNET", "images")
    ds = FGNetDataset(data_file, size=128, channels=3)
    train, test = ds.get_split(0.2)

    base_model = get_model()
    
    regress = Regression(base_model, (128,128,3))
    regress_history = regress.train(train, batch_size=128, epochs=5, save_file="regression_model")
    with open("regression_history.json", w) as file:
        json.dump(regress_history.history, file)
    mse = regress.validate(test)
    with open("regression_mse.txt", mode='w') as file:
        file.write(mse)
        
    
    hl_model = HL_model(base_model, bins, (128,128,3))
    hl_histroy = hl_model.train(train, batch_size=128, epochs=5, save_file="hl_model")
    with open("hl_history.json", w) as file:
        json.dump(hl_history.history, file)
    mse = hl_model.validate(test)
    with open("hl_mse.txt", mode='w') as file:
        file.write(mse)
    
    
    classification_model = Classification(base_model, bins, (128,128,3))
    classification_history = classification_model.train(train, batch_size=128, epochs=5, save_file="classification_model")
    with open("classification_history.json", w) as file:
        json.dump(classification_history.history, file)
    mse = classification_model.validate(test)
    with open("classification_mse.txt", mode='w') as file:
        file.write(mse)
    
    
    return

    
if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)