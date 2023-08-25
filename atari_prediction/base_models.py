"""Base models for the atari prediction problem."""

from tensorflow import keras
from keras import layers


def value_network():
    """Small convolutional value network.
    Based on the example from sample_test.py

    Returns: a keras model that accepts stacked image inputs and outputs a feature layer.
    """
    return keras.models.Sequential([
        layers.Permute((2, 3, 1)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 8, 4),
        layers.LeakyReLU(),
        layers.Conv2D(64, 4, 2),
        layers.LeakyReLU(),
        layers.Conv2D(64, 3, 1),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(512),
        layers.LeakyReLU()
        ])


def large_model(image_size = (84, 84), num_images=4, output_size=1, output_activation=None, dropout=0.5):
    """Larger convolutional neural network.

    Params:
        image_size - the image dimensions (w, h)
        num_images - the number of images in each stack
        output_size - the size of the output layer
        output_activation - the activation for the output layer
        dropout - the dropout used before each of the final three dense layers
    
    Returns: a keras model that accepts stacked image inputs with configurable outputs
    """
    inputs = layers.Input(shape=(num_images, image_size[0], image_size[1]))
    x = layers.Rescaling(scale=1./255)(inputs)
    x = layers.Permute((2, 3, 1))(x)
    x = layers.Conv2D(filters = 64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters = 64, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = layers.BatchNormalization(axis=[1,2,3])(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units = output_size, activation = output_activation)(x)
    model = keras.Model(inputs=inputs, outputs = outputs)
    return model
