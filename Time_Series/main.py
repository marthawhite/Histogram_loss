import tensorflow as tf
from tensorflow import keras
import pandas as pd
from experiment.models import HLGaussian, Regression
from Time_Series.time_series_transformer import get_model
import json
import sys


def reshape(n, chans):
    return lambda x: tf.transpose(tf.reshape(x, (-1, n, chans)), [0, 2, 1])


def get_data(in_file, seq_len, pred_len, chans=7):
    df = pd.read_csv(in_file)
    df = df.drop("date", axis=1)
    df = tf.convert_to_tensor(df, dtype=tf.float32)
    dmin = tf.reduce_min(df, axis=0)
    dmax = tf.reduce_max(df, axis=0)
    dif = dmax - dmin
    scale = tf.where(dif == 0, 1., dif)
    df = (df - dmin) / scale
    x = keras.utils.timeseries_dataset_from_array(df[:-pred_len], None, sequence_length=seq_len)
    y = keras.utils.timeseries_dataset_from_array(df[seq_len:], None, sequence_length=pred_len).map(reshape(pred_len, chans))
    return tf.data.Dataset.zip((x, y))


def get_bins(n_bins, pad_ratio, sig_ratio):
    """Return the histogram bins given the HL parameters.
    
    Params:
        n_bins - the number of bins to create (includes padding)
        pad_ratio - the number of sigma of padding to use on each side 
        sig_ratio - the ratio of sigma to bin width

    Returns: 
        borders - a Tensor of n_bins + 1 bin borders
        sigma - the sigma to use for HL-Gaussian
    """
    bin_width = 1 / (n_bins - 2 * sig_ratio * pad_ratio)
    pad_width = sig_ratio * pad_ratio * bin_width
    borders = tf.linspace(-pad_width, 1 + pad_width, n_bins + 1)
    sigma = bin_width * sig_ratio
    return borders, sigma


def main(data_path):
    pred_len = 72
    seq_len = 336
    epochs = 20
    sig_ratio = 2.
    pad_ratio = 3.
    n_bins = 25
    chans = 7
    head_size = 256
    n_heads = 4
    features = 128
    data = get_data(data_path, seq_len, pred_len, chans)
    metrics = ["mse", "mae"]

    borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio)
    borders = tf.expand_dims(borders, -1)
    borders = tf.expand_dims(borders, -1)

    base = get_model(data.element_spec[0].shape[1:], head_size, n_heads, features)

    hlg = HLGaussian(base, borders, sigma, out_shape=(pred_len,))    
    hlg.compile("adam", None, metrics)
    hist = hlg.fit(data, epochs=epochs, verbose=2)
    with open(f"HL_transformer.json", "w") as file:
        json.dump(hist.history, file)

    base = get_model(data.element_spec[0].shape[1:], head_size, n_heads, features)

    reg = Regression(base, out_shape=(pred_len,))    
    reg.compile("adam", "mse", metrics)
    hist = reg.fit(data, epochs=epochs, verbose=2)
    with open(f"Reg_transformer.json", "w") as file:
        json.dump(hist.history, file)


if __name__ == "__main__":
    data_path = sys.argv[1]
    main(data_path)
