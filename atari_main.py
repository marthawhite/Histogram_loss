import tensorflow as tf
from tensorflow import keras
from keras import layers
from experiment.models import HLGaussian, Regression
import sys
import json
from experiment.RL_dataset import get_dataset
import numpy as np

def test_model():
    return keras.models.Sequential([
        layers.Permute((2, 3, 1)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 8, 4, activation="relu"),
        layers.Conv2D(64, 4, 2, activation="relu"),
        layers.Conv2D(64, 3, 1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu")
        ])


def get_model(image_size = (84, 84), num_images=4, output_size=1, output_activation=None):
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
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(axis=1)(x)
    outputs = layers.Dense(units = output_size, activation = output_activation)(x)
    model = keras.Model(inputs=inputs, outputs = outputs)
    return model
    
    
def main(action_file, returns_file):
    keras.utils.set_random_seed(1)
    n_epochs = 50
    batch_size = 32
    bin_width = 0.015
    borders = tf.range(-0.25,1.25, bin_width, tf.float32)
    num_batches_train = 10000
    num_batches_test = 1000
    sig_ratio = 2.
    dropout = 0.5
    learning_rate = 1e-3
    metrics = ["mse", "mae"]
    
    ds = get_dataset(action_file, returns_file).shuffle(32*32).batch(batch_size).prefetch(1)
    
    train = ds.take(num_batches_train)
    test = ds.take(num_batches_test)
    
    hl_gaussian = HLGaussian(test_model(), borders, sig_ratio * bin_width, dropout)
    hl_gaussian.compile(optimizer=keras.optimizers.Adam(learning_rate), metrics=metrics)
    hl_gaussian_history = hl_gaussian.fit(x=train, epochs=n_epochs, validation_data=test, verbose=2)
    with open(f"hlg.json", "w") as file:
        json.dump(hl_gaussian_history.history, file)

    # Save samples to examine after
    for x, y in ds.take(1):
        out = hl_gaussian.get_hist(x, training=False)
        np.save(f"hists.npy", out.numpy())
        np.save(f"y.npy", y.numpy())
    
    ds = get_dataset(action_file, returns_file).shuffle(32*32).batch(batch_size).prefetch(1)
    
    train = ds.take(num_batches_train)
    test = ds.take(num_batches_test)
    
    regression = Regression(test_model())
    regression.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=metrics)
    regression_history = regression.fit(x=train, epochs=n_epochs, validation_data=test, verbose=2)
    with open("reg.json", "w") as file:
        json.dump(regression_history.history, file)
    print(regression.predict(ds.take(1), verbose=2))
    
    
    
if __name__ == "__main__":
    action_file = sys.argv[1]
    returns_file = sys.argv[2]
    main(action_file, returns_file)
    
