import tensorflow as tf
from tensorflow import keras
import pandas as pd
from experiment.models import HLGaussian, Regression
from Time_Series.time_series_transformer import get_model
import json
import sys


def reshape(n, chans):
    return lambda x: tf.transpose(tf.reshape(x, (n, chans)), [1, 0])


def get_time_series_dataset(filename, drop=[], seq_len=720, train_len=20, pred_len=720, test_size=0.2, batch_size=64, chans=7):
    # test_size is the portion of the dataset to use as test data must be between 0 and 1
    df = pd.read_csv(filename)
    df = df.drop(drop, axis = 1)
    df = tf.convert_to_tensor(df, dtype=tf.float32)
    dmin = tf.reduce_min(df, axis=0)
    dmax = tf.reduce_max(df, axis=0)
    dif = dmax - dmin
    scale = tf.where(dif == 0, 1., dif)
    df = (df - dmin) / scale
    n = df.shape[0] 
    split = round((1-test_size)*n)
    data = df
    train = data[:split]
    test = data[split:]
    inputs = train[:-(train_len)]
    target = train[seq_len:]
    x_train = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y_train = keras.utils.timeseries_dataset_from_array(target, None, train_len, batch_size=None).map(reshape(pred_len, chans))
    ds_train = tf.data.Dataset.zip((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    inputs = test[:-(pred_len)]
    targets = test[seq_len:]
    x = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y = keras.utils.timeseries_dataset_from_array(targets, None, pred_len, batch_size=None).map(reshape(pred_len, chans))
    ds_test = tf.data.Dataset.zip((x,y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test


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
    epochs = 60
    sig_ratio = 2.
    pad_ratio = 3.
    n_bins = 25
    chans = 7
    head_size = 256
    n_heads = 4
    features = 128
    test_ratio = 0.1
    batch_size = 64
    drop = "date"
    train, test = get_time_series_dataset(data_path, drop, seq_len, pred_len, pred_len, test_ratio, batch_size, chans)
    metrics = ["mse", "mae"]

    borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio)
    borders = tf.expand_dims(borders, -1)
    borders = tf.expand_dims(borders, -1)

    shape = train.element_spec[0].shape[1:]

    base = get_model(shape, head_size, n_heads, features)

    hlg = HLGaussian(base, borders, sigma, out_shape=(pred_len,))    
    hlg.compile("adam", None, metrics)
    hist = hlg.fit(train, epochs=epochs, verbose=2, validation_data=test)
    with open(f"HL_transformer.json", "w") as file:
        json.dump(hist.history, file)

    base = get_model(shape, head_size, n_heads, features)

    reg = Regression(base, out_shape=(pred_len,))    
    reg.compile("adam", "mse", metrics)
    hist = reg.fit(train, epochs=epochs, verbose=2, validation_data=test)
    with open(f"Reg_transformer.json", "w") as file:
        json.dump(hist.history, file)


if __name__ == "__main__":
    data_path = sys.argv[1]
    main(data_path)
