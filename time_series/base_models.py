"""Module containing base models for time series experiments."""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from experiment.multidense import MultiDense
from time_series.autoformer import SeriesDecomp


def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    """Pre-norm Transformer encoder block.

    Self-attention with key_dim = d_model // num_heads, then a position-wise
    feed-forward network (width ff_dim, GELU), each with a residual connection
    and layer normalization.

    Params:
        inputs - tensor of shape (batch, seq_len, d_model)
        d_model - model dimension
        num_heads - number of attention heads
        ff_dim - feed-forward hidden dimension (typically 4 * d_model)
        dropout - dropout rate

    Returns: tensor of shape (batch, seq_len, d_model)
    """
    attn = layers.MultiHeadAttention(
        key_dim=d_model // num_heads, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


def transformer(input_shape, pred_len, d_model=128, n_heads=8, ff_dim=512,
                n_blocks=3, dropout=0.1):
    """Vanilla Transformer encoder for forecasting (Vaswani et al., 2017).

    Hyperparameter conventions follow the original Transformer (and the ETT
    transformers in Informer/Autoformer): key_dim = d_model // n_heads and
    feed-forward width ff_dim = 4 * d_model. Input channels are projected to
    d_model, a learnable positional embedding is added, n_blocks encoder layers
    are applied, then a linear head maps the time axis (seq_len) to pred_len.

    (batch, seq_len, channels) -> (batch, pred_len, d_model)

    Params:
        input_shape - (seq_len, channels)
        pred_len - number of future timesteps to predict
        d_model - model hidden dimension
        n_heads - number of attention heads (key_dim = d_model // n_heads)
        ff_dim - feed-forward hidden dimension (set to 4 * d_model)
        n_blocks - number of encoder blocks
        dropout - dropout rate

    Returns: a Keras model to use as the base
    """
    seq_len = input_shape[0]
    inputs = keras.Input(shape=input_shape)

    x = layers.Dense(d_model)(inputs)                       # project channels -> d_model
    positions = tf.range(seq_len)
    x = x + layers.Embedding(seq_len, d_model)(positions)   # learnable positional embedding
    x = layers.Dropout(dropout)(x)

    for _ in range(n_blocks):
        x = transformer_encoder(x, d_model, n_heads, ff_dim, dropout)

    # Linear projection head: (batch, seq_len, d_model) -> (batch, pred_len, d_model)
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(pred_len)(x)
    x = layers.Permute((2, 1))(x)
    return keras.Model(inputs, x)


def linear(chans, seq_len, pred_len):
    """Channel-independent, target-only Linear (Zeng et al., 2023, "LTSF-Linear").

    (batch, seq_len, channels) -> (batch, pred_len, 1)
    """
    inputs = keras.Input(shape=(seq_len, chans))
    x = layers.Lambda(lambda t: t[:, :, -1:])(inputs)   # target channel -> (batch, seq_len, 1)
    x = layers.Permute((2, 1))(x)                        # (batch, 1, seq_len)
    x = layers.Dense(pred_len)(x)                        # seq_len -> pred_len
    x = layers.Permute((2, 1))(x)                        # (batch, pred_len, 1)
    return keras.Model(inputs, x)


def independent_dense(chans, seq_len):
    """Return a Keras model to form the base of a 3 independent dense layer model.
    Permutes the input channels and timesteps, but does not modify the data.

    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the form
        (batchsize, channels, timesteps)

    Params:
        chans - the number of channels
        seq_len - the number of input timesteps

    Returns: a Keras model to use as the base
    """
    return keras.models.Sequential([
        keras.layers.Reshape((seq_len, chans)),
        keras.layers.Permute([2, 1]),
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
        chans - the number of channels
        seq_len - the number of input timesteps

    Returns: a Keras model to use as the base
    """
    return keras.models.Sequential([
        keras.layers.Reshape((seq_len * chans,)),
        keras.layers.Dense(seq_len * chans, activation="relu"),
        keras.layers.Dense(seq_len * chans, activation="relu"),
        keras.layers.Reshape((chans, seq_len))
    ])


def lstm_encdec(width, pred_len, n_layers, drop, input_shape):
    """Return an LSTM encoder base model with a linear projection head.
    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the shape
        (batchsize, pred_len, width)

    Params:
        width - the size of the feature dimension for the LSTM layers
        pred_len - number of future timesteps to predict
        n_layers - the number of stacked LSTM layers
        drop - the dropout rate
        input_shape - the shape of the input; does not include batch dimension

    Returns: a keras model to use as the base
    """
    inputs = keras.Input(input_shape)
    x = inputs
    for i in range(n_layers):
        x = layers.LSTM(width, return_sequences=True, dropout=drop)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Linear projection head: map encoded sequence to prediction length
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(pred_len)(x)
    x = layers.Permute((2, 1))(x)
    return keras.Model(inputs, x)


def gru_linear(width, pred_len, n_layers, drop, input_shape):
    """Return a GRU encoder base model with a linear projection head.

    Identical in structure to lstm_encdec but uses GRU layers instead of LSTM.

    Takes input of the form:
        (batchsize, timesteps, channels)
    and produces output of the shape
        (batchsize, pred_len, width)

    Params:
        width - the size of the feature dimension for the GRU layers
        pred_len - number of future timesteps to predict
        n_layers - the number of stacked GRU layers
        drop - the dropout rate
        input_shape - the shape of the input; does not include batch dimension

    Returns: a keras model to use as the base
    """
    inputs = keras.Input(input_shape)
    x = inputs
    for i in range(n_layers):
        x = layers.GRU(width, return_sequences=True, dropout=drop)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Linear projection head: map encoded sequence to prediction length
    x = layers.Permute((2, 1))(x)
    x = layers.Dense(pred_len)(x)
    x = layers.Permute((2, 1))(x)
    return keras.Model(inputs, x)
