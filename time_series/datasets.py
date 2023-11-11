"""Module for loading time series data"""

from experiment.dataset import Dataset
import tensorflow as tf
import pandas as pd
from tensorflow import keras


def reshape(T, chans):
    """Return a function that permutes timesteps and channels.
    Apply to UNBATCHED data via tf.data.Dataset.map.

    Params:
        T - the number of timesteps
        chans - the number of channels

    Returns: a function to map to data
    """
    return lambda x: tf.transpose(tf.reshape(x, (T, chans)), [1, 0])


def get_ETT_split(data, filename, seq_len):
    """Create data splits according to Informer and LTSF-Linear.
    Creates 12-4-4 month train-val-test split where each month is 30 days 

    Sources: 
        Informer https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py
        LTSF-Linear https://github.com/cure-lab/LTSF-Linear/blob/main/data_provider/data_loader.py
    
    Params:
        data - the Tensor containing the raw ETT data
        filename - the name of the input file; one of ETT{h1/h2/m1/m2}.csv
        seq_len - the length of the input sequences

    Returns: train, val, test - the split dataset
    """
    periods = 1
    if filename[-6] == "m":
        periods = 4
    samples_per_month = 30 * 24 * periods  # 30 days
    train_len = 12 * samples_per_month  # 12 months
    test_len = 4 * samples_per_month
    val_end = train_len + test_len
    train = data[:train_len]
    val = data[train_len - seq_len:val_end]
    test = data[val_end - seq_len:val_end + test_len]
    return train, val, test


def get_time_series_dataset(filename, drop=[], seq_len=720, batch_size=64, chans=7, input_target_offset=0,eps=1e-08,univariate=True):
    """Return the train/test split for a CSV time series dataset.
    Uses 12-4 month split to be comparable to standard 12-4-4 train-val-test for ETTh datasets.
    
    Params:
        filename - the name of the CSV file containing the data
        drop - the names of the columns to drop
            Note: All non-numeric data should be dropped
        seq_len - the length of the input sequences
        train_len - the length of the training target sequences
        pred_len - the length of the prediction target sequences
        test_size - the proportion of samples to use for testing
            WARNING: NOT USED IN CURRENT IMPLEMENTATION
        batch_size - the size of the data batches
        chans - the number of channels (features) in the data
        input_target_offset - the number of timesteps between the last input timestep and the first output timestep

    Returns: ds_train, ds_test, dmin, dmax
        ds_train - a tf.data.Dataset containing (x, y) tuples of inputs and targets for training
        ds_test - a tf.data.Dataset containing (x, y) tuples of inputs and targets for testing
        dmin - a Tensor with the minimum values for each feature in the training split
        dmax - a Tensor with the maximum values for each feature in the training split
    """
    
    # test_size is the portion of the dataset to use as test data must be between 0 and 1
    df = pd.read_csv(filename)
    df = df.drop(drop, axis = 1)
    df = tf.convert_to_tensor(df, dtype=tf.float32)

    mu = tf.reduce_mean(df, axis=0)
    sig = tf.math.reduce_std(df, axis=0)
    scale = sig + eps
    df = (df - mu) / scale
    df_inputs = df[:-(seq_len+input_target_offset)]
    df_targets = df[(seq_len+input_target_offset):]
    xs = keras.utils.timeseries_dataset_from_array(df_inputs, None, seq_len, batch_size=None)
    ys = keras.utils.timeseries_dataset_from_array(df_targets[:,-1], None, 1, batch_size=None)
    ds = tf.data.Dataset.zip((xs, ys))
    
    periods = 1
    if filename[-6] == "m":
        periods = 4
    samples_per_month = 30 * 24 * periods  # 30 days
    total_samples = samples_per_month * 24
    train_len = 12 * samples_per_month  # 12 months
    test_len = 4 * samples_per_month
    
    train,test = tf.keras.utils.split_dataset(ds, left_size=train_len/total_samples,right_size=test_len/total_samples,shuffle=True,seed=0)
    train = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test = test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dmin = tf.reduce_min(df[:,-1], axis=0)
    dmax = tf.reduce_max(df[:,-1], axis=0)
    return train, test, dmin, dmax
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
