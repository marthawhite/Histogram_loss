"""Module containing the run code for time series experiments.

Usage: 
    python main.py data_path

Params:
    data_path - path to the input data file
"""

import tensorflow as tf
from tensorflow import keras
from experiment.models import HLGaussian, Regression
from time_series.base_models import transformer, linear, lstm_encdec
import json
from experiment.bins import get_bins
from time_series.datasets import get_time_series_dataset


def main():
    """Run the time series experiment.
    
    Params:
        data_path - path to the data file
    """
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    pred_len = 336
    seq_len = 96
    epochs = 100
    sig_ratio = 2.
    pad_ratio = 3.
    n_bins = 100
    chans = 7
    # head_size = 512
    # n_heads = 8
    # features = 128
    layers = 2
    width = 512
    test_ratio = 0.25
    batch_size = 32
    drop = "date"
    metrics = ["mse", "mae"]
    lr = 1e-4

    for dataset in datasets:
        keras.utils.set_random_seed(1)
        data_path = f"{dataset}.csv"
        train, test, dmin, dmax = get_time_series_dataset(data_path, drop, seq_len, pred_len, pred_len, test_ratio, batch_size, chans)

        borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio, dmin, dmax)
        borders = tf.expand_dims(borders, -1)
        sigma = tf.expand_dims(sigma, -1)

        shape = train.element_spec[0].shape[1:]

        #base = transformer(shape, head_size, n_heads, features)
        #base = linear(chans, seq_len)
        base = lstm_encdec(width, layers, 0.5, shape)

        hlg = HLGaussian(base, borders, sigma, out_shape=(pred_len,))    
        hlg.compile(keras.optimizers.Adam(lr), None, metrics)
        hist = hlg.fit(train, epochs=epochs, verbose=2, validation_data=test)
        with open(f"HL_lstm_{dataset}.json", "w") as file:
            json.dump(hist.history, file)

        # #base = transformer(shape, head_size, n_heads, features)
        # base = linear(chans, seq_len)
        # #base = lstm_encdec(width, layers, 0.5, shape)

        # reg = Regression(base, out_shape=(pred_len,))    
        # reg.compile(keras.optimizers.Adam(lr), "mse", metrics)
        # hist = reg.fit(train, epochs=epochs, verbose=2, validation_data=test)
        # with open(f"Reg_linear_{dataset}.json", "w") as file:
        #     json.dump(hist.history, file)


if __name__ == "__main__":
    main()
