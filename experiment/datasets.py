"""Dataset classes that produce Tensorflow datasets ready to use in Keras models."""

import tensorflow as tf
import os
import pandas as pd


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
            x = self.preprocess(x)
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

    def preprocess(self, ds):
        """Preprocess the data.
        
        Params:
            args - one unpacked element of the dataset
        """
        return ds

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


class ImageDataset(Dataset):
    """A dataset of images loaded from files.
    
    Params:
        size - the width (and height) of the image after resizing
        channels - 1 or 3; the number of color channels
        **kwargs - arguments for the dataset class; includes buffer_size, batch_size, prefetch 
    """

    def __init__(self, size=128, channels=3, **kwargs) -> None:
        self.size = size
        self.channels = channels
        super().__init__(**kwargs)

    def preprocess(self, x):
        """Process the image data from a file.
        
        Params:
            x - the filename

        Returns: a tuple of the image tensor and label
        """
        return x.map(lambda x: self.parse_image(x))

    def parse_image(self, filename):
        """Read the image from a file and convert to a tensor and determine the label.

        Params:
            filename - the path to the image file
        
        Returns image, label - the (size, size, channels) tensor for the image and its label
        """
        label = self.parse_label(filename)

        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=self.channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.size, self.size])
        return image, label

    def parse_label(self, filename):
        """Parse the label from a filename."""
        pass


class MegaAgeDataset(ImageDataset):
    """MegaAge Asian age estimation dataset.
    Contains 40000 images with ages from 0 to 70
    Source: http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/

    Params:
        path - the path to the image directories;
            should have train, test or train_aligned, test_aligned directories available
        aligned - use aligned image directories if True, unaligned otherwise
        **kwargs - arguments for the dataset class; includes buffer_size, batch_size, prefetch
    """
    
    def __init__(self, path, aligned=True, **kwargs) -> None:
        self.path = path
        self.aligned = aligned
        super().__init__(**kwargs)

    def load(self):
        """Load the images."""
        if self.aligned:
            dirs = "train_aligned", "test_aligned"
        else:
            dirs = "train", "test"
            
        train_glob = os.path.join(self.path, dirs[0], "*")
        test_glob = os.path.join(self.path, dirs[1], "*")

        train_ds = tf.data.Dataset.list_files(train_glob, shuffle=False)
        test_ds = tf.data.Dataset.list_files(test_glob, shuffle=False)

        y_train, y_test = self.load_labels(len(train_ds), len(test_ds))
        self.n_train = y_train.shape[0]
        self.labels = tf.concat((y_train, y_test), axis=0)
        self.test_dir = dirs[1]
        self.train = train_ds
        self.test = test_ds

    def split(self, data, val_ratio, test_ratio):
        """Return a default train-test split if val_ratio is None."""
        if val_ratio is None:
            return self.train, self.test
        else:
            return super().split(data, val_ratio, test_ratio)
        
    def get_data(self):
        """Return the combined train and test data."""
        return self.train.concatenate(self.test)

    def parse_label(self, filename):
        """Parse the image label from its filename.
        
        Params:
            filename - the path to the image file

        Returns: the age of the person in the image
        """
        path = tf.strings.split(filename, os.sep)
        
        img_dir = path[-2]
        file = path[-1]
        index = tf.strings.to_number(tf.strings.split(file, ".")[0], out_type=tf.int32)
        index = tf.where(tf.equal(img_dir, "test"), index + self.n_train, index)
        return self.labels[index - 1]

    def load_labels(self, len_train, len_test):
        """Load the train and test labels from text files.
        
        Params:
            len_train - the number of train samples
            len_test - the number of test samples

        Returns: train, test - tensors containing train and test labels indexed by image number
        """
        path = os.path.join(self.path, "list", "train_age.txt")
        train_ds = tf.data.TextLineDataset(path).batch(len_train)
        train = tf.strings.to_number(train_ds.get_single_element())

        path = os.path.join(self.path, "list", "test_age.txt")
        test_ds = tf.data.TextLineDataset(path).batch(len_test)
        test = tf.strings.to_number(test_ds.get_single_element())
        return train, test


class FGNetDataset(ImageDataset):
    """FGNet age estimation dataset.
    Contains 1002 images of 82 subjects from ages 0-70.
    Source: https://yanweifu.github.io/FG_NET_data/

    Params:
        path - the path to the directory containing the images
        **kwargs - arguments for the dataset class; includes buffer_size, batch_size, prefetch
    """

    def __init__(self, path, **kwargs) -> None:
        self.path = path
        super().__init__(**kwargs)
        
    def parse_label(self, filename):
        """Parse the age label from the 4-6th characters in the filename.
        
        Params:
            filename - the path to the image file
        
        Returns: label - the integer age of the person in the image
        """
        parts = tf.strings.split(filename, os.sep)
        label = tf.strings.to_number(tf.strings.substr(parts[-1], 4, 2))
        return label
    
    def load(self):
        """Load the files as a tf.data.Dataset"""
        glob = os.path.join(self.path, "*")
        list_ds = tf.data.Dataset.list_files(glob, shuffle=False)
        self.data = list_ds
    
    def get_data(self):
        """Return the loaded data."""
        return self.data
    

class UTKFaceDataset(ImageDataset):
    """UTKFace age estimation dataset.
    Contains 20000 images with ages from 0 to 116.
    Source: https://susanqq.github.io/UTKFace/
    """

    def __init__(self, path, **kwargs):
        self.path = path
        super().__init__(**kwargs)
        
    def parse_label(self, filename):
        """Return the age of the person in the image from the filename.
        
        Params:
            filename - the path to the image file

        Returns: the age of the person in the image
        """
        parts = tf.strings.split(filename, os.sep)
        label = tf.strings.to_number(tf.strings.split(parts[-1], "_")[0])
        return label
    
    def load(self):
        """Load the data from files."""
        glob = os.path.join(self.path, "*")
        list_ds = tf.data.Dataset.list_files(glob, shuffle=False)
        self.data = list_ds

    def get_data(self):
        """Return the loaded dataset"""
        return self.data
    