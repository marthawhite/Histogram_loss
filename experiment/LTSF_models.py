import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_models(L=256):
    return keras.Input(shape=(L,))
     