"""Module containing base models for time series experiments."""

from tensorflow import keras
from keras import layers
from experiment.multidense import MultiDense



def transformer(input_shape, head_size, num_heads, feature_dims):
    """Return a Keras model implementing a transformer.
    Source: https://keras.io/examples/timeseries/timeseries_classification_transformer/

    Takes input of the form
        (batchsize, timesteps, channels)
    and produces outputs with shape
        (batchsize, channels, features)
    
    Params:
        input_shape - the shape of the input data; does not include batch size
            Should be of the form (timesteps, channels)
        head_size - the size of the self-attention heads
        num_heads - the number of self-attention heads
        feature_dims - the dimension of the output features for each channel
        depth - the number of self-attention blocks
    
    Returns: a Keras model to use as the base
    """
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


def independent_dense(chans, seq_len):
    """Return a Keras model to form the base of a 3 independent dense layer model.
    Permutes the input channels and timesteps, but does not modify the data.
    
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the form
        (batchsize, channels, timesteps)

    Params:
        channels - the number of channels
        seq_len - the number of input timesteps
    
    Returns: a Keras model to use as the base
    """
    return keras.model.Sequential([
        keras.layers.Reshape((seq_len, chans)),
        keras.layers.Permute([2,1]),
        MultiDense(shape=(seq_len,)),
        keras.layers.ReLU(),
        MultiDense(shape=(seq_len,)),
        keras.layers.ReLU()
    ])


def dependent_dense(chans, seq_len):
    """Return a Keras model to form the base of a 3 dense layer model.
    Permutes the input channels and timesteps, but does not modify the data.
    
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the form
        (batchsize, channels, timesteps)

    Params:
        channels - the number of channels
        seq_len - the number of input timesteps
    
    Returns: a Keras model to use as the base
    """
    return keras.model.Sequential({
        keras.layers.Reshape((seq_len*chans)),
        keras.layers.Dense(seq_len*chans, activation="relu"),
        keras.layers.Dense(seq_len*chans, activation="relu"),
        keras.layers.Reshape((chans, seq_len))
    })


def linear(chans, seq_len):
    """Return a Keras model to form the base of a linear model.
    Permutes the input channels and timesteps, but does not modify the data.
    
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the form
        (batchsize, channels, timesteps)

    Params:
        channels - the number of channels
        seq_len - the number of input timesteps
    
    Returns: a Keras model to use as the base
    """
    return keras.models.Sequential([
        keras.layers.Reshape((seq_len, chans)),
        keras.layers.Permute([2, 1])
    ])


def lstm_encdec(width, n_layers, drop, input_shape):
    """Return an LSTM encoder-decoder base model.
    
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the shape
        (batchsize, width)

    Params:
        width - the size of the feature dimension for the LSTM layer and dense layers
        n_layers - the number of linear blocks in the encoder and decoder
        drop - the dropout rate to use in the encoder and decoder blocks
        input_shape - the shape of the input; does not include batch dimension
    
    Returns: a keras model to use as the base
    """
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
