from experiment.dataset import Dataset
import tensorflow as tf
import pandas as pd


class CSVDataset(Dataset):
    """A dataset from a CSV file.
    
    Params:
        path - the path to the file
        targets - the column name(s) containing the targets to predict
        drop - the column name(s) to exclude from the data
            Note: Any non-numeric columns must be dropped!
        **kwargs - arguments for the dataset class; includes buffer_size, batch_size, prefetch
    """

    def __init__(self, path, targets, drop=[], **kwargs) -> None:
        self.path = path
        self.targets = targets
        self.drop = drop
        super().__init__(**kwargs)

    def load(self):
        """Read the input data from the file."""
        df = pd.read_csv(self.path)
        df = df.drop(self.drop, axis=1)
        x = df.drop(self.targets, axis=1)
        y = df[self.targets]
        ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x, dtype=tf.float32),tf.convert_to_tensor(y, dtype=tf.float32)))
        self.ds = ds

    def get_data(self):
        """Return the loaded dataset"""
        return self.ds
