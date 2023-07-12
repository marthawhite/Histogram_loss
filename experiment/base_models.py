from tensorflow import keras
from keras import layers


def value_network():
    return keras.models.Sequential([
        layers.Permute((2, 3, 1)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 8, 4, activation="relu"),
        layers.Conv2D(64, 4, 2, activation="relu"),
        layers.Conv2D(64, 3, 1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu")
        ])


def large_model(image_size = (84, 84), num_images=4, output_size=1, output_activation=None, dropout=0.5):
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
