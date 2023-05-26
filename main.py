import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from models import HL_model, Regression
from experiment.datasets import FGNetDataset
import os




def main():
    bins = np.arange(-10, 80)
    path = os.path.join("data", "FGNET", "images")
    ds = FGNetDataset(path, size=128, channels=3)
    train, test = ds.get_split(0.2)
    
    
    base_model = keras.applications.Xception(
        include_top=False,
        weights=None,
        input_tensor=layers.Input(shape=(128,128,3)),
        pooling="avg",
    )
    
    
    regress = Regression(base_model, (128,128,3))
    regress.train(train, batch_size=128, epochs=5, save_file="regression_model")
    mse = regress.validate(test)
    with open("regression_mse", mode='w') as file:
        file.write(mse)
        
    
    hl_model = HL_model(base_model, bins, (128,128,3))
    hl_model.train(train, batch_size=128, epochs=5, save_file="hl_model")
    mse = regress.validate(test)
    with open("regression_mse", mode='w') as file:
        file.write(mse)
    
    return

    
if __name__ == "__main__":
    main()