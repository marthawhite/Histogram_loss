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


def _make_windows(data, seq_len, input_target_offset, pred_len):
    """Create aligned (input, target) sliding windows from a contiguous data block.

    For window i:
        x = data[i : i + seq_len]                                       (all channels)
        y = data[i + seq_len + offset : i + seq_len + offset + pred_len] (last channel)

    The data block must be at least seq_len + input_target_offset + pred_len rows long
    to produce at least one window.

    Params:
        data - a 2-D Tensor of shape (timesteps, channels), already normalised
        seq_len - input window length
        input_target_offset - gap between end of input and start of target
        pred_len - number of target timesteps

    Returns: a tf.data.Dataset of (x, y) pairs (unbatched)
    """
    total_window = seq_len + input_target_offset + pred_len
    n_windows = tf.shape(data)[0] - total_window + 1
    # inputs start at index 0; targets start at index seq_len + offset
    df_inputs = data[:-(input_target_offset + pred_len)]
    df_targets = data[(seq_len + input_target_offset):]
    xs = keras.utils.timeseries_dataset_from_array(df_inputs, None, seq_len, batch_size=None)
    ys = keras.utils.timeseries_dataset_from_array(df_targets[:, -1], None, pred_len, batch_size=None)
    return tf.data.Dataset.zip((xs, ys))


def get_time_series_dataset(filename, drop=[], seq_len=720, batch_size=64, chans=7,
                            input_target_offset=0, eps=1e-08, univariate=True, pred_len=1):
    """Return train/val/test datasets for an ETT CSV file.

    Uses the standard 12-4-4 month split (Informer / LTSF-Linear convention).
    Normalisation statistics and histogram bin ranges are computed from the
    training split only to avoid data leakage.

    Each split is backed up by seq_len + input_target_offset rows so that the
    first sliding window can look back into the preceding period, following
    the same convention used in Informer and LTSF-Linear.

    Params:
        filename - CSV file name (e.g. "ETTh1.csv")
        drop - column name(s) to drop (e.g. "date")
        seq_len - input window length
        batch_size - batch size
        chans - number of channels (features) in the data
        input_target_offset - gap between end of input and start of target
        eps - small constant for numerical stability in normalisation
        univariate - whether to predict a single target variable
        pred_len - number of future timesteps to predict

    Returns: ds_train, ds_val, ds_test, dmin, dmax
    """
    df = pd.read_csv(filename)
    df = df.drop(drop, axis=1)
    data = tf.convert_to_tensor(df, dtype=tf.float32)

    periods = 1
    if filename[-6] == "m":
        periods = 4
    samples_per_month = 30 * 24 * periods  # 30 days
    train_end = 12 * samples_per_month       # 12 months training
    val_end = train_end + 4 * samples_per_month  # 4 months validation
    test_end = val_end + 4 * samples_per_month   # 4 months test

    train_raw = data[:train_end]
    mu = tf.reduce_mean(train_raw, axis=0)
    sig = tf.math.reduce_std(train_raw, axis=0)
    scale = sig + eps
    data = (data - mu) / scale

    train_targets = data[:train_end, -1]
    dmin = tf.reduce_min(train_targets)
    dmax = tf.reduce_max(train_targets)

    lookback = seq_len + input_target_offset

    train_block = data[:train_end]
    val_block = data[train_end - lookback : val_end]
    test_block = data[val_end - lookback : test_end]

    ds_train = _make_windows(train_block, seq_len, input_target_offset, pred_len)
    ds_val = _make_windows(val_block, seq_len, input_target_offset, pred_len)
    ds_test = _make_windows(test_block, seq_len, input_target_offset, pred_len)

    # Shuffle training data only (within training set)
    train_size = tf.data.experimental.cardinality(ds_train).numpy()
    ds_train = ds_train.shuffle(train_size, seed=0)

    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, dmin, dmax
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
