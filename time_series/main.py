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


def main(base_model, loss):
    """Run the time series experiment.
    
    Params:
        base_model(str):  Name of the base model
        loss(str): Name of loss, either HL or L2
        
    General Params:
        datasets: Array of the names of the datasets to train and test the model on
        seq_len: Number of previous timesteps to be given as input to the model (this is the same for all channels)
        pred_len: Number of future timesteps the model will predict (this is the same for all channels)
        epochs: Number of epochs to train for
        sig_ratio: Sigma ratio of the discretized histogram transform
        pad_ratio: Padding ratio of the discretized histogram transform
        n_bins: Number of bins in the discretized histogram
        chans: Number of variables in the dataset
        test_ratio: Ratio of test data to train data
        batch_size: Batch Size for the training
        drop: Name of column(s) that contain data that should not be used as inputs (for multiple columns use a list of strings)
        metrics: Metrics for evaluating train and test performance
        lr: Learning rate
        input_target_offset: Number of steps between the last input time step and the first target time step
        
    Model Specific Params:
        Transformer:
            head_size: Size of head for key and query
            n_heads: Number of attention heads
            features: Number of feature dimensions
        LSTM:
            layers: Number of layers in the encoder and decoder
            width: Width of the layers in the encoder and decoder
        
        
    """
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    pred_len = 336
    seq_len = 336
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
    input_target_offset = 0

    for dataset in datasets:
        keras.utils.set_random_seed(1)
        data_path = f"{dataset}.csv"
        train, test, dmin, dmax = get_time_series_dataset(data_path, drop, seq_len, pred_len, pred_len, test_ratio, batch_size, chans, input_target_offset)

        borders, sigma = get_bins(n_bins, pad_ratio, sig_ratio, dmin, dmax)
        borders = tf.expand_dims(borders, -1)
        sigma = tf.expand_dims(sigma, -1)

        shape = train.element_spec[0].shape[1:]

        out_shape = (pred_len,)
        if base_model == "transformer":
            base = transformer(shape, head_size, n_heads, features)
        elif base_model == "LSTM":
            out_shape = (chans, pred_len)
            base = lstm_encdec(width, layers, 0.5, shape)
        elif base_model == "linear":
            base = linear(chans, seq_len)
        elif base_model == "independent_dense":
            base = independent_dense(chans, seq_len)
        else:
            base = dependent_dense(chans, seq_len)
            
        if loss == "HL":
            hlg = HLGaussian(base, borders, sigma, out_shape=out_shape)    
            hlg.compile(keras.optimizers.Adam(lr), None, metrics)
            hist = hlg.fit(train, epochs=epochs, verbose=2, validation_data=test)
            with open(f"HL_{dataset}_{base_model}.json", "w") as file:
                json.dump(hist.history, file)
        else:
            reg = Regression(base, out_shape=out_shape)    
            reg.compile(keras.optimizers.Adam(lr), "mse", metrics)
            hist = reg.fit(train, epochs=epochs, verbose=2, validation_data=test)
            with open(f"L2_{dataset}_{base_model}.json", "w") as file:
                json.dump(hist.history, file)
        


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
