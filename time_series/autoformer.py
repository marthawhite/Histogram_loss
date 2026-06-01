"""
Ported from https://github.com/thuml/Autoformer
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import math


class MovingAvg(layers.Layer):
    """Moving average block to extract the trend from a time series."""

    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2

    def call(self, x):
        front = tf.repeat(x[:, :1, :], self.pad, axis=1)
        end = tf.repeat(x[:, -1:, :], self.pad, axis=1)
        x_padded = tf.concat([front, x, end], axis=1)
        x_avg = tf.keras.backend.pool2d(
            tf.expand_dims(x_padded, -1),
            pool_size=(self.kernel_size, 1),
            strides=(1, 1),
            padding="valid",
            pool_mode="avg",
        )
        return tf.squeeze(x_avg, -1)


class SeriesDecomp(layers.Layer):
    """Decompose a series into trend (moving average) and seasonal."""

    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.moving_avg = MovingAvg(kernel_size)

    def call(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(layers.Layer):
    """Auto-Correlation mechanism with period-based dependency discovery
    and time-delay aggregation.

    Replaces standard self-attention with O(L log L) complexity.
    """

    def __init__(self, factor=1, dropout=0.05, seq_len=96,
                 query_len=None, key_len=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.dropout_layer = layers.Dropout(dropout)
        # Pre-compute top_k from known sequence length (avoids runtime shape issues)
        self._top_k = max(1, int(factor * math.log(seq_len)))
        # Pre-determine length alignment mode from known static lengths
        q_len = query_len or seq_len
        k_len = key_len or seq_len
        if q_len > k_len:
            self._align = "pad"
        elif q_len < k_len:
            self._align = "slice"
        else:
            self._align = "none"

    def call(self, queries, keys, values, training=None):
        # queries/keys/values: (batch, length, heads, d_k)
        L = tf.shape(queries)[1]

        if self._align == "pad":
            S = tf.shape(values)[1]
            B = tf.shape(queries)[0]
            H = tf.shape(queries)[2]
            D = tf.shape(values)[3]
            E = tf.shape(queries)[3]
            pad_v = tf.zeros(tf.stack([B, L - S, H, D]))
            pad_k = tf.zeros(tf.stack([B, L - S, H, E]))
            values = tf.concat([values, pad_v], axis=1)
            keys = tf.concat([keys, pad_k], axis=1)
        elif self._align == "slice":
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # (batch, length, heads, d) -> (batch, heads, d, length)
        q = tf.transpose(queries, [0, 2, 3, 1])
        k = tf.transpose(keys, [0, 2, 3, 1])
        v = tf.transpose(values, [0, 2, 3, 1])

        # Period-based dependencies via FFT cross-correlation
        q_fft = tf.signal.rfft(tf.cast(q, tf.float32))
        k_fft = tf.signal.rfft(tf.cast(k, tf.float32))
        corr_fft = q_fft * tf.math.conj(k_fft)
        corr = tf.signal.irfft(corr_fft)  # (batch, heads, d, length)

        # Find top-k delays
        top_k = self._top_k

        # Average correlation across heads and channels -> (batch, length)
        mean_corr = tf.reduce_mean(tf.reduce_mean(corr, axis=1), axis=1)

        # Batch-norm style (shared top-k indices) — works for both
        # training and inference.
        global_corr = tf.reduce_mean(mean_corr, axis=0)  # (length,)
        _, indices = tf.math.top_k(global_corr, k=top_k)  # (top_k,)
        weights = tf.gather(mean_corr, indices, axis=-1)  # (batch, top_k)
        weights = tf.nn.softmax(weights, axis=-1)

        # Aggregate by rolling values at each delay
        delays_agg = tf.zeros_like(v)
        for i in range(top_k):
            shift = indices[i]
            pattern = tf.roll(v, shift=-shift, axis=-1)
            w = weights[:, i]  # (batch,)
            w = tf.reshape(w, [-1, 1, 1, 1])
            delays_agg = delays_agg + pattern * w

        # (batch, heads, d, length) -> (batch, length, heads, d)
        out = tf.transpose(delays_agg, [0, 3, 1, 2])
        return out

    def compute_output_shape(self, queries_shape, keys_shape, values_shape):
        return (queries_shape[0], queries_shape[1],
                queries_shape[2], queries_shape[3])


class AutoCorrelationLayer(layers.Layer):
    """Multi-head Auto-Correlation layer with Q/K/V projections."""

    def __init__(self, d_model, n_heads, factor=1, dropout=0.05,
                 seq_len=96, query_len=None, key_len=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        d_keys = d_model // n_heads
        self.d_keys = d_keys
        self.query_proj = layers.Dense(d_keys * n_heads)
        self.key_proj = layers.Dense(d_keys * n_heads)
        self.value_proj = layers.Dense(d_keys * n_heads)
        self.out_proj = layers.Dense(d_model)
        self.correlation = AutoCorrelation(
            factor=factor, dropout=dropout, seq_len=seq_len,
            query_len=query_len, key_len=key_len)

    def call(self, queries, keys, values, training=None):
        H = self.n_heads
        dk = self.d_keys
        # Use tf.shape for runtime-dynamic batch & length dims
        B = tf.shape(queries)[0]
        L = tf.shape(queries)[1]
        S = tf.shape(keys)[1]

        q = tf.reshape(self.query_proj(queries), [B, L, H, dk])
        k = tf.reshape(self.key_proj(keys), [B, S, H, dk])
        v = tf.reshape(self.value_proj(values), [B, S, H, dk])

        out = self.correlation(q, k, v, training=training)
        out = tf.reshape(out, [B, L, H * dk])
        return self.out_proj(out)

    def compute_output_shape(self, queries_shape, keys_shape, values_shape):
        return (queries_shape[0], queries_shape[1], self.d_model)


class TokenEmbedding(layers.Layer):
    """Conv1D token embedding (kernel_size=3, circular padding)."""

    def __init__(self, c_in, d_model, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(d_model, kernel_size=3, padding="valid",
                                  use_bias=False)
        self.c_in = c_in
        self.d_model = d_model

    def call(self, x):
        # Circular padding: wrap last element to front and first to end
        x_padded = tf.concat([x[:, -1:, :], x, x[:, :1, :]], axis=1)
        return self.conv(x_padded)


class AutoformerEncoderLayer(layers.Layer):
    """Autoformer encoder layer: Auto-Correlation + decomp + FF + decomp."""

    def __init__(self, d_model, n_heads, d_ff, moving_avg=25, factor=1,
                 dropout=0.05, activation="relu", seq_len=96, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.auto_corr = AutoCorrelationLayer(d_model, n_heads, factor, dropout, seq_len=seq_len)
        self.conv1 = layers.Conv1D(d_ff, kernel_size=1, use_bias=False)
        self.conv2 = layers.Conv1D(d_model, kernel_size=1, use_bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = layers.Dropout(dropout)
        self.activation_fn = tf.nn.relu if activation == "relu" else tf.nn.gelu

    def call(self, x, training=None):
        # Auto-Correlation + residual
        attn_out = self.auto_corr(x, x, x, training=training)
        x = x + self.dropout(attn_out, training=training)
        x, _ = self.decomp1(x)

        # Feedforward + residual
        y = self.dropout(self.activation_fn(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        seasonal, _ = self.decomp2(x + y)
        return seasonal

    def compute_output_shape(self, input_shape):
        return input_shape


class AutoformerDecoderLayer(layers.Layer):
    """Autoformer decoder layer with progressive decomposition.

    Each layer produces a seasonal output and accumulates trend residuals.
    """

    def __init__(self, d_model, n_heads, c_out, d_ff, moving_avg=25,
                 factor=1, dropout=0.05, activation="relu",
                 dec_seq_len=144, enc_seq_len=96, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.c_out = c_out
        self.self_corr = AutoCorrelationLayer(
            d_model, n_heads, factor, dropout, seq_len=dec_seq_len)
        self.cross_corr = AutoCorrelationLayer(
            d_model, n_heads, factor, dropout, seq_len=dec_seq_len,
            query_len=dec_seq_len, key_len=enc_seq_len)
        self.conv1 = layers.Conv1D(d_ff, kernel_size=1, use_bias=False)
        self.conv2 = layers.Conv1D(d_model, kernel_size=1, use_bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = layers.Dropout(dropout)
        # Trend projection: Conv1D(d_model -> c_out, kernel=3, circular padding)
        self.trend_proj = layers.Conv1D(c_out, kernel_size=3, padding="valid",
                                        use_bias=False)
        self.activation_fn = tf.nn.relu if activation == "relu" else tf.nn.gelu

    def _circular_conv(self, x):
        """Apply trend projection with circular padding."""
        x_padded = tf.concat([x[:, -1:, :], x, x[:, :1, :]], axis=1)
        return self.trend_proj(x_padded)

    def call(self, x, cross, training=None):
        # Self Auto-Correlation + decomp
        x = x + self.dropout(
            self.self_corr(x, x, x, training=training), training=training)
        x, trend1 = self.decomp1(x)

        # Cross Auto-Correlation + decomp
        x = x + self.dropout(
            self.cross_corr(x, cross, cross, training=training), training=training)
        x, trend2 = self.decomp2(x)

        # Feedforward + decomp
        y = self.dropout(self.activation_fn(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        x, trend3 = self.decomp3(x + y)

        # Accumulate and project trend
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self._circular_conv(residual_trend)
        return x, residual_trend

    def compute_output_shape(self, x_shape, cross_shape):
        return (x_shape, (x_shape[0], x_shape[1], self.c_out))


class SeasonalLayerNorm(layers.Layer):
    """LayerNorm that subtracts the temporal mean (keeps seasonal zero-mean)."""

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x_hat = self.ln(x)
        bias = tf.reduce_mean(x_hat, axis=1, keepdims=True)
        return x_hat - bias


def autoformer(input_shape, pred_len, d_model=512, n_heads=8, d_ff=2048,
               e_layers=2, d_layers=1, factor=1, moving_avg=25, dropout=0.05,
               activation="relu", c_out=None):
    """Build an Autoformer base model in Keras Functional API.

    Follows the original implementation: encoder processes the full input,
    decoder receives seasonal/trend initialization and refines predictions
    through progressive decomposition with Auto-Correlation.

    Takes input of the form:
        (batchsize, seq_len, n_channels)
    and produces output of the form:
        (batchsize, pred_len, d_model)

    Params:
        input_shape - (seq_len, n_channels)
        pred_len - number of future timesteps to predict
        d_model - model hidden dimension
        n_heads - number of Auto-Correlation heads
        d_ff - feedforward hidden dimension
        e_layers - number of encoder layers
        d_layers - number of decoder layers
        factor - top-k factor for Auto-Correlation (k = factor * log(L))
        moving_avg - kernel size for moving average decomposition
        dropout - dropout rate
        activation - "relu" or "gelu"
        c_out - output channels for trend projection (defaults to n_channels)

    Returns: a Keras model to use as the base
    """
    seq_len = input_shape[0]
    n_channels = input_shape[1]
    if c_out is None:
        c_out = n_channels
    label_len = seq_len // 2

    inputs = keras.Input(shape=input_shape)  # (batch, seq_len, n_channels)

    decomp_init = SeriesDecomp(moving_avg, name="decomp_init")
    seasonal_init_full, trend_init_full = decomp_init(inputs)

    enc_embed = TokenEmbedding(n_channels, d_model, name="enc_embedding")
    enc_out = enc_embed(inputs)
    enc_out = layers.Dropout(dropout)(enc_out)

    for i in range(e_layers):
        enc_out = AutoformerEncoderLayer(
            d_model, n_heads, d_ff, moving_avg, factor, dropout, activation,
            seq_len=seq_len, name=f"enc_layer_{i}"
        )(enc_out)

    enc_out = SeasonalLayerNorm(d_model, name="enc_norm")(enc_out)

    seasonal_start = layers.Cropping1D(
        cropping=(seq_len - label_len, 0))(seasonal_init_full)
    zeros_seasonal = layers.Lambda(
        lambda x: tf.zeros((tf.shape(x)[0], pred_len, n_channels))
    )(inputs)
    seasonal_dec_input = layers.Concatenate(axis=1)(
        [seasonal_start, zeros_seasonal])

    trend_start = layers.Cropping1D(
        cropping=(seq_len - label_len, 0))(trend_init_full)
    mean_init = layers.Lambda(
        lambda x: tf.repeat(
            tf.reduce_mean(x, axis=1, keepdims=True), pred_len, axis=1)
    )(inputs)
    trend_dec = layers.Concatenate(axis=1)([trend_start, mean_init])

    # Project initial trend from n_channels to c_out so it matches
    # the residual_trend dimensions from decoder layers
    if n_channels != c_out:
        trend_dec = layers.Dense(c_out, use_bias=False, name="trend_init_proj")(trend_dec)

    dec_embed = TokenEmbedding(n_channels, d_model, name="dec_embedding")
    dec_out = dec_embed(seasonal_dec_input)
    dec_out = layers.Dropout(dropout)(dec_out)

    dec_seq_len = label_len + pred_len
    for i in range(d_layers):
        dec_layer = AutoformerDecoderLayer(
            d_model, n_heads, c_out, d_ff, moving_avg, factor, dropout,
            activation, dec_seq_len=dec_seq_len, enc_seq_len=seq_len,
            name=f"dec_layer_{i}")
        dec_out, residual_trend = dec_layer(dec_out, enc_out)
        trend_dec = trend_dec + residual_trend

    dec_out = SeasonalLayerNorm(d_model, name="dec_norm")(dec_out)

    # Project seasonal part to c_out dimensions
    dec_seasonal = layers.Dense(c_out, name="dec_projection")(dec_out)

    # Final output = seasonal + trend, take last pred_len steps
    final_out = dec_seasonal + trend_dec
    output = layers.Cropping1D(cropping=(label_len, 0))(final_out)

    return keras.Model(inputs, output)
