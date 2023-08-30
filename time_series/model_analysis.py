"""Module containing the run code for time series experiments.

Usage: 
    python main.py data_path

Params:
    data_path - path to the input data file
"""

import tensorflow as tf
from tensorflow import keras
from experiment.models import HLGaussian, Regression
from time_series.base_models import transformer, linear, lstm_encdec, independent_dense, dependent_dense
import json
from experiment.bins import get_bins
from time_series.datasets import get_time_series_dataset
import numpy as np
import sys


def main(base_model, loss):
    """Run the time series experiment.
    
    Params:
        data_path - path to the data file
    """
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    pred_len = 1
    seq_len = 96
    epochs = 100
    sig_ratio = 2.
    pad_ratio = 3.
    n_bins = 100
    chans = 7
    head_size = 512
    n_heads = 8
    features = 128
    layers = 2
    width = 512
    test_ratio = 0.25
    batch_size = 32
    drop = "date"
    metrics = ["mse", "mae"]
    lr = 1e-3
    input_target_offset = 10

    for dataset in datasets:
        keras.utils.set_random_seed(1)
        data_path = f"{dataset}.csv"
        train, test, dmin, dmax = get_time_series_dataset(data_path, drop, seq_len, pred_len, pred_len, test_ratio, batch_size, chans, input_target_offset)

        borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio, dmin, dmax)
        borders = tf.expand_dims(borders, -1)
        sigma = tf.expand_dims(sigma, -1)

        shape = train.element_spec[0].shape[1:]
        
        test_inputs = test.map(lambda x,y: x)
        test_targets = test.map(lambda x,y: y).unbatch()
        elements = np.array([element for element in test_targets.as_numpy_iterator()])
        np.save(f"{dataset}_targets", elements)
        
        
        if base_model == "transformer":
            base = transformer(shape, head_size, n_heads, features)
        elif base_model = "LSTM":
            base = lstm_encdec(width, layers, 0.5, shape)
        elif base_model = "linear":
            base = linear(chans, seq_len)
        elif base_model = "independent_dense":
            base = independent_dense(chans, seq_len)
        else:
            base = dependent_dense(chans, seq_len)
            
        if loss = "HL":
            hlg = HLGaussian(base, borders, sigma, out_shape=(pred_len,))    
            hlg.compile(keras.optimizers.Adam(lr), None, metrics)
            hist = hlg.fit(train, epochs=epochs, verbose=2, validation_data=test)
            with open(f"HL_{dataset}_{base_model}.json", "w") as file:
                json.dump(hist.history, file)
            predictions = hlg.predict(test_inputs)
            np.save(f"{dataset}_{base_model}_HL", predictions)
        else:
            reg = Regression(base, out_shape=(pred_len,))    
            reg.compile(keras.optimizers.Adam(lr), "mse", metrics)
            hist = reg.fit(train, epochs=epochs, verbose=2, validation_data=test)
            with open(f"Reg_{dataset}_{base_model}.json", "w") as file:
                json.dump(hist.history, file)
            predictions = reg.predict(test_inputs)
            np.save(f"{dataset}_{base_model}_reg", predictions)
        

        #base = transformer(shape, head_size, n_heads, features)
        #base = linear(chans, seq_len)
        #base = lstm_encdec(width, layers, 0.5, shape)

        #reg = Regression(base, out_shape=(pred_len,))    
        #reg.compile(keras.optimizers.Adam(lr), "mse", metrics)
        #hist = reg.fit(train, epochs=epochs, verbose=2, validation_data=test)
        #with open(f"Reg_transformer_{dataset}.json", "w") as file:
        #    json.dump(hist.history, file)
        #predictions = reg.predict(test_inputs)
        #np.save(f"{dataset}_reg", predictions)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
