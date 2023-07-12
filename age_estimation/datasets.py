import tensorflow as tf
import os
from experiment.dataset import Dataset


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
        return self.parse_image(x)

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
    