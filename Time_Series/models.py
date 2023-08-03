from tensorflow import keras
from keras import layers


def transformer(input_shape, head_size, num_heads, feature_dims):
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


def linear_model(chans, seq_len):
    return keras.models.Sequential([
        keras.layers.Reshape((seq_len, chans)),
        keras.layers.Permute([2, 1])
    ])


def lstm_model(width, n_layers, drop, input_shape):
    inputs = keras.Input(input_shape)
    x = inputs
    for i in range(n_layers):
        x = layers.TimeDistributed(layers.Dense(width, activation="relu"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Dropout(drop))(x)
    x = layers.LSTM(width)(x)
    for i in range(n_layers):
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
        x = layers.Dense(width, activation="relu")(x)
    return keras.Model(inputs, x)
