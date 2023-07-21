from experiment.dataset import Dataset
import tensorflow as tf
import pandas as pd


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
