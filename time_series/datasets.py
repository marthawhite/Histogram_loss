"""Module for loading time series data"""

from experiment.dataset import Dataset
import tensorflow as tf
import pandas as pd
from tensorflow import keras


def reshape(n, chans):
    return lambda x: tf.transpose(tf.reshape(x, (n, chans)), [1, 0])


def get_time_series_dataset(filename, drop=[], seq_len=720, train_len=20, pred_len=720, test_size=0.2, batch_size=64, chans=7):
    # test_size is the portion of the dataset to use as test data must be between 0 and 1
    df = pd.read_csv(filename)
    df = df.drop(drop, axis = 1)
    df = tf.convert_to_tensor(df, dtype=tf.float32)
    mu = tf.reduce_mean(df, axis=0)
    sig = tf.math.reduce_std(df, axis=0)
    scale = tf.where(sig == 0, 1., sig)
    df = (df - mu) / scale
    
    hours_per_month = 730
    train_months = 12
    test_months = 4
    test_start = -(test_months * hours_per_month + seq_len + pred_len - 1)
    train_start = test_start - (train_months * hours_per_month + seq_len + train_len - 1)

    n = df.shape[0] 
    split = round((1-test_size)*n)
    data = df
    train = data[train_start:test_start]
    test = data[test_start:]
    inputs = train[:-(train_len)]
    target = train[seq_len:]
    dmin = tf.reduce_min(train, axis=0)
    dmax = tf.reduce_max(train, axis=0)
    x_train = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y_train = keras.utils.timeseries_dataset_from_array(target, None, train_len, batch_size=None).map(reshape(pred_len, chans))
    ds_train = tf.data.Dataset.zip((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    inputs = test[:-(pred_len)]
    targets = test[seq_len:]

    x = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y = keras.utils.timeseries_dataset_from_array(targets, None, pred_len, batch_size=None).map(reshape(pred_len, chans))
    ds_test = tf.data.Dataset.zip((x,y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test, dmin, dmax


class TSDataset(Dataset):
    """A dataset of time-series data read from a CSV file.
    
    Params:
        path - a path to the csv file
        seq_len - the window length used as input
        pred_len - the window length that is predicted
        targets - the column name(s) used as targets if mode is 'S' or 'MS'
        drop - the column name(s) to exclude from the data
            Note: Any non-numeric columns must be dropped!
        mode - one of 'S', 'M', or 'MS'; determines the structure of the x and y features
            'S' -> learning and predicting the targets only
            'MS' -> learning on all columns and predicting targets
            'M' -> learning and predicting all columns
        overlap - the number of timesteps of overlap between the input and prediction windows;
            negative allows a gap between the input and prediction windows
        **kwargs - arguments for the dataset class; includes buffer_size, batch_size, prefetch
    """

    def __init__(self, path, seq_len, pred_len, targets=None, drop=[], mode='M', overlap=0, **kwargs) -> None:
        self.path = path
        self.targets = targets
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.overlap = overlap
        self.drop = drop
        super().__init__(**kwargs)

    def load(self):
        """Read the data from the input file and convert it into time windows."""
        df = pd.read_csv(self.path)
        df = df.drop(self.drop, axis=1)

        self.n = len(df) - (self.seq_len + self.pred_len - self.overlap) + 1

        if self.mode == 'S':
            df = df[self.targets]

        tensor = tf.convert_to_tensor(df, dtype=tf.float32)
        base = tf.data.Dataset.from_tensor_slices(tensor)
        x = base.window(self.seq_len, shift=1).flat_map(lambda x: x.batch(self.seq_len, drop_remainder=True)).take(self.n)
        
        if self.mode == 'MS':
            df = df[self.targets]
            tensor = tf.convert_to_tensor(df, dtype=tf.float32)
            base = tf.data.Dataset.from_tensor_slices(tensor)

        y = base.skip(self.seq_len - self.overlap).window(self.pred_len, shift=1).flat_map(lambda x: x.batch(self.pred_len, drop_remainder=True))
        self.ds = tf.data.Dataset.zip((x, y))

    def get_data(self):
        """Return the loaded data"""
        return self.ds
    
    def __len__(self):
        """Return the length of the dataset"""
        return self.n