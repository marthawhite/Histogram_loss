from tensorflow import keras


def get_models(L=256):
    return keras.Input(shape=(L,))
     