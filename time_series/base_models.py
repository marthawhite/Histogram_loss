"""Module containing base models for time series experiments."""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from experiment.multidense import MultiDense


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    #Source: https://keras.io/examples/timeseries/timeseries_classification_transformer/
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def transformer_encoder_v2(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    """Transformer encoder block with pre-norm, GELU activation, and Dense feedforward.

    Params:
        inputs - tensor of shape (batch, seq_len, d_model)
        d_model - model dimension
        num_heads - number of attention heads
        ff_dim - feedforward hidden dimension
        dropout - dropout rate

    Returns: tensor of shape (batch, seq_len, d_model)
    """
    # Self-attention with residual
    attn = layers.MultiHeadAttention(
        key_dim=d_model // num_heads, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    # Feedforward with residual
    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    out = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return out


def transformer_large(input_shape, pred_len, d_model=512, n_heads=8, ff_dim=2048, n_layers=6, dropout=0.1):
    """
    Takes input of the form
        (batchsize, seq_len, n_channels)
    and produces output of the form
        (batchsize, pred_len, d_model)

    The final d_model dimension is projected to scalar (Regression) or n_bins
    (HL-Gaussian) by the model wrapper's MultiDense layer.

    Params:
        input_shape - (seq_len, n_channels)
        pred_len - number of future timesteps to predict
        d_model - transformer hidden dimension
        n_heads - number of attention heads
        ff_dim - feedforward hidden dimension
        n_layers - number of transformer encoder blocks
        dropout - dropout rate

    Returns: a Keras model to use as the base
    """
    seq_len = input_shape[0]
    inputs = keras.Input(shape=input_shape)

    # Project input channels to model dimension
    x = layers.Dense(d_model)(inputs)  # (seq_len, d_model)

    # Learnable positional embedding
    positions = tf.range(seq_len)
    pos_embedding = layers.Embedding(seq_len, d_model)(positions)
    x = x + pos_embedding
    x = layers.Dropout(dropout)(x)

    # Transformer encoder stack
    for _ in range(n_layers):
        x = transformer_encoder_v2(x, d_model, n_heads, ff_dim, dropout)

    # Linear projection head: map encoded sequence to prediction length
    # (batch, seq_len, d_model) -> (batch, d_model, seq_len) -> Dense -> (batch, d_model, pred_len) -> (batch, pred_len, d_model)
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(pred_len)(x)
    x = layers.Permute((2, 1))(x)

    return keras.Model(inputs, x)

def transformer(input_shape, pred_len, head_size, num_heads, feature_dims):
    """Return a Keras model implementing a transformer.
    Source: https://keras.io/examples/timeseries/timeseries_classification_transformer/

    Takes input of the form
        (batchsize, seq_len, channels)
    and produces outputs with shape
        (batchsize, pred_len, channels)
    
    Params:
        input_shape - the shape of the input data; does not include batch size
            Should be of the form (seq_len, channels)
        pred_len - number of future timesteps to predict
        head_size - the size of the self-attention heads
        num_heads - the number of self-attention heads
        feature_dims - the dimension of the output features for each channel
    
    Returns: a Keras model to use as the base
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(3):
        x = transformer_encoder(x, head_size, num_heads, feature_dims)
    # (batch, seq_len, channels) -> (batch, pred_len, channels)
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(pred_len)(x)
    x = layers.Permute((2, 1))(x)
    return keras.Model(inputs, x)


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
    return keras.models.Sequential([
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
    return keras.models.Sequential([
        keras.layers.Reshape((seq_len*chans,)),
        keras.layers.Dense(seq_len*chans, activation="relu"),
        keras.layers.Dense(seq_len*chans, activation="relu"),
        keras.layers.Reshape((chans, seq_len))
    ])


def linear(chans, seq_len, pred_len):
    """Return a Keras model to form the base of a linear model.
    
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the form
        (batchsize, pred_len, channels)

    Params:
        chans - the number of channels
        seq_len - the number of input timesteps
        pred_len - number of future timesteps to predict
    
    Returns: a Keras model to use as the base
    """
    return keras.models.Sequential([
        keras.layers.Reshape((seq_len * chans,)),
        keras.layers.Dense(pred_len * chans, activation="relu"),
        keras.layers.Reshape((pred_len, chans))
    ])


def lstm_encdec(width, pred_len, n_layers, drop, input_shape):
    """Return an LSTM encoder-decoder base model.
    
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the shape
        (batchsize, pred_len, width)

    Params:
        width - the size of the feature dimension for the LSTM layer and dense layers
        pred_len - number of future timesteps to predict
        n_layers - the number of linear blocks in the encoder and decoder
        drop - the dropout rate to use in the encoder and decoder blocks
        input_shape - the shape of the input; does not include batch dimension
    
    Returns: a keras model to use as the base
    """
    inputs = keras.Input(input_shape)
    x = inputs
    x = layers.LSTM(width)(x)
    x = layers.RepeatVector(pred_len)(x)
    return keras.Model(inputs, x)
