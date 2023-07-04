import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from experiment.new_models import HLGaussian, HLOneBin, Regression
import os
import sys
import json
from experiment.RL_dataset import get_dataset
    
    
def get_model(image_size = (1, 84, 84), num_images=4, output_size=1, output_activation=None):
    inputs = layers.Input(shape=((image_size[0]*num_images, image_size[1], image_size[2])))
    x = layers.Conv2D(filters = 64, kernel_size=(3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters = 64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(axis=1)(x)
    outputs = layers.Dense(units = output_size, activation = output_activation)(x)
    model = keras.Model(inputs=inputs, outputs = outputs)
    return model
    
    
def main(action_file, returns_file):
    n_epochs = 8
    batch_size = 32
    borders = tf.range(-0.25,1.25, 0.015, tf.float32)
    num_batches_train = 1000
    num_batches_test = 100
    
    
    ds = get_dataset(action_file, returns_file).shuffle(32).batch(batch_size).prefetch(1)
    
    train = ds.take(num_batches_train)
    test = ds.take(num_batches_test)
    
    hl_gaussian = HLGaussian(get_model(output_size=128), borders, 1.0, 0.0)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("hl_gaussian_history.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)
        
    
    ds = get_dataset(returns_file).shuffle(32).batch(batch_size).prefetch(1)
    
    train = ds.take(num_batches_train)
    test = ds.take(num_batches_test)
    
    regression = Regression(get_model(output_size=128))
    regression.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    regression_history = regression.fit(x=train, epochs=n_epochs, validation_data=test)
    with open("regression_history.json", "w") as file:
        json.dump(regression_history.history, file)
    
    
    
if __name__ == "__main__":
    action_file = sys.argv[1]
    returns_file = sys.argv[2]
    main(action_file, returns_file)
    