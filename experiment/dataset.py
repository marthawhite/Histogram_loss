import tensorflow as tf


class Dataset:
    """Base dataset class. Provides an interface to get pre-batched and shuffled
    train-(val)-test splits of data to be passed into keras model training API methods.
    
    Params:
        buffer_size - the size of the shuffle buffer; None for size of dataset
        batch_size - the number of samples per mini-batch
        prefetch - the number of mini-batches to preload
    """

    def __init__(self, buffer_size=None, batch_size=32, prefetch=tf.data.AUTOTUNE) -> None:
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.buf = buffer_size
        self.load()

    def prepare(self, splits):
        """Prepare the data for use. Shuffle, preprocess, batch, and prefetch.
        
        Params:
            splits - list of datasets to prepare
        
        Returns: data - a list of the prepared datasets
        """
        data = []
        for x in splits:
            x = self.shuffle(x)
            x = x.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            x = x.batch(self.batch_size).prefetch(self.prefetch)
            data.append(x)
        return data

    def load(self):
        """Read the data from input files/directories."""
        pass

    def shuffle(self, data, reshuffle=True):
        """Shuffle the data according to the buffer size.
        
        Params:
            data - the tf.data.Dataset to shuffle
            reshuffle - True if the data should be reshuffled each iteration
        
        Returns: a tf.data.Dataset with the shuffled data
        """
        if self.buf is None:
            buf = len(self)
        else:
            buf = self.buf
        return data.shuffle(buf, reshuffle_each_iteration=reshuffle)

    def preprocess(self, *args):
        """Preprocess the data.
        
        Params:
            args - one unpacked element of the dataset
        """
        return args

    def get_data(self):
        """Return the data that has been loaded."""
        pass

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.get_data())

    def get_split(self, val_ratio, test_ratio=None, shuffle=False):
        """Create a train-val-(test) split from the data.
        
        Params:
            val_ratio - the size of the validation split;
                proportional to dataset size if < 1, else number of samples
            test_ratio - the size of the test split; None if train-val split only
                proportional to dataset size if < 1, else number of samples
            shuffle - True if the data should be shuffled before splitting, False otherwise
        
        Returns: a list of tf.data.Datasets containing the splits ready for use in a model
        """
        data = self.get_data()
        
        if shuffle:
            data = self.shuffle(data, False)

        splits = self.split(data, val_ratio, test_ratio)        

        return self.prepare(splits)
    
    def split(self, data, val_ratio, test_ratio):
        """Split the dataset into train-val-(test).
        
        Params:
            data - the tf.data.Dataset to split
            val_ratio - the size of the validation split;
                proportional to dataset size if < 1, else number of samples
            test_ratio - the size of the test split; None if train-val split only
                proportional to dataset size if < 1, else number of samples
        
        Returns: a tuple of tf.data.Datasets containing the splits
        """
        if test_ratio is not None:
            return self.three_split(data, val_ratio, test_ratio)
        else:
            return self.two_split(data, val_ratio)
    
    def two_split(self, data, test_ratio):
        """Split the dataset into train-test.
        
        Params:
            data - the tf.data.Dataset to split
            test_ratio - the size of the test split;
                proportional to dataset size if < 1, else number of samples
        
        Returns: train, test - the tf.data.Datasets containing the splits
        """
        test_len = self.get_num(test_ratio)
        train_len = len(self) - test_len

        train = data.take(train_len)
        test = data.skip(train_len).take(test_len)
        return train, test

    def get_num(self, size):
        """Return the number of samples given a size input.
        
        Params:
            size - a proportion or number of samples

        Returns: the number of samples
        """
        if size >= 1:
            return size
        else:
            return int(len(self) * size)

    def three_split(self, data, val_ratio, test_ratio):
        """Split the dataset into train-val-test.
        
        Params:
            data - the tf.data.Dataset to split
            val_ratio - the size of the validation split;
                proportional to dataset size if < 1, else number of samples
            test_ratio - the size of the test split;
                proportional to dataset size if < 1, else number of samples
        
        Returns: train, val, test - the tf.data.Datasets containing the splits
        """
        test_len = self.get_num(test_ratio)
        val_len = self.get_num(val_ratio)
        train_len = len(data) - val_len - test_len

        train = data.take(train_len)
        val = data.skip(train_len).take(val_len)
        test = data.skip(train_len + val_len)
        return train, val, test
