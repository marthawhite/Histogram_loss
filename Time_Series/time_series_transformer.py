import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_model(input_shape, head_size, num_heads, feature_dims):
    values = input_shape[-1]
    inputs = keras.Input(shape=input_shape)
    res = inputs
    for i in range(5):
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(res, res)
        res = inputs + x 
        x = layers.BatchNormalization()(res)
        x = layers.Conv1D(filters=values, kernel_size=1)(x)
        res = x + res
    # (batchsize, timesteps, values)
    x = layers.Conv1D(filters=values*feature_dims, kernel_size=1)(res)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Reshape((values, feature_dims))(x)
    return keras.Model(inputs, outputs)


