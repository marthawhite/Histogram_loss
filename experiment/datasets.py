import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

class Dataset:
    """Base dataset class."""

    def __init__(self, batch_size=32, prefetch=1) -> None:
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.load()

    def prepare(self, *args):
        return [x.batch(self.batch_size).prefetch(self.prefetch) for x in args]

    def load(self):
        pass

    def get_data(self):
        pass

    def __len__(self):
        return len(self.get_data())

    def input_shape(self):
        return self.ds.element_spec[0].shape

    def get_split(self, test_ratio, shuffle=True):
        """Return a train-test split for the given test_ratio.
        
        Params:
            self - the Dataset object
            test_ratio - float in range [0,1] indicating the proportion of test samples

        Returns: a tuple (train, test) of tf.data.Dataset
        """

        data = self.get_data()
        if shuffle:
            data = data.shuffle(len(self), reshuffle_each_iteration=False)

        train_len = int(len(self) * (1-test_ratio))
        return self.prepare(data.take(train_len).shuffle(train_len), data.skip(train_len).shuffle(len(self) - train_len))

    def three_split(self, val_ratio, test_ratio):
        data = self.get_data()
        train_len = int(len(self) * (1 - test_ratio - val_ratio))
        val_len = int(len(self) * val_ratio)
        train = data.take(train_len)
        val = data.skip(train_len).take(val_len)
        test = data.skip(train_len + val_len)
        return self.prepare(train, val, test)

class CSVDataset(Dataset):

    def __init__(self, path, targets) -> None:
        self.path = path
        self.targets = targets
        super().__init__()

    def load(self):
        df = pd.read_csv(self.path)
        x = df.drop(self.targets, axis=1)
        y = df[self.targets]
        ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x, dtype=tf.float32),tf.convert_to_tensor(y, dtype=tf.float32)))
        self.ds = ds

    def get_data(self):
        return self.ds
    

class TSDataset(Dataset):

    def __init__(self, path, targets, seq_len, pred_len, mode='S', scale=True, overlap=0) -> None:
        self.path = path
        self.targets = targets
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.scale = scale
        self.overlap = overlap
        super().__init__()

    def load(self):
        df = pd.read_csv(self.path)
        df = df.drop("date", axis=1)
        self.n = len(df) - (self.seq_len + self.pred_len - self.overlap) + 1
        if self.mode == 'S':
            df = df[self.targets]
        tensor = tf.convert_to_tensor(df, dtype=tf.float32)
        if self.scale:
            tensor = (tensor - tf.math.reduce_mean(tensor, axis=0)) / tf.math.reduce_std(tensor, axis=0)
        base = tf.data.Dataset.from_tensor_slices(tensor)
        #print(tensor.shape)
        x = base.window(self.seq_len, shift=1).flat_map(lambda x:x.batch(self.seq_len, drop_remainder=True))#.take(self.n)
        if self.mode == 'MS':
            df = df[self.targets]
            tensor = tf.convert_to_tensor(df, dtype=tf.float32)
            if self.scale:
                tensor = (tensor - tf.math.reduce_mean(tensor, axis=0)) / tf.math.reduce_std(tensor, axis=0)
            base = tf.data.Dataset.from_tensor_slices(tensor)
        y = base.skip(self.seq_len - self.overlap).window(self.pred_len, shift=1).flat_map(lambda x: x.batch(self.pred_len, drop_remainder=True))
        self.ds = tf.data.Dataset.zip((x, y))

    def get_data(self):
        return self.ds
    
    def __len__(self):
        return self.n


class ImageDataset(Dataset):

    def __init__(self, size=128, channels=3, **kwargs) -> None:
        self.size = size
        self.channels = channels
        super().__init__(**kwargs)

    def parse_image(self, filename):
        label = self.parse_label(filename)

        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=self.channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.size, self.size])
        return image, label

    def parse_label(self, filename):
        pass


class MegaAgeDataset(ImageDataset):
    
    def __init__(self, path, size, channels, batch_size=32, aligned=True) -> None:
        self.path = path
        self.aligned = aligned
        super().__init__(size, channels, batch_size=batch_size)

    def load(self):
        if self.aligned:
            dirs = "train_aligned", "test_aligned"
        else:
            dirs = "train", "test"
            
        train_glob = os.path.join(self.path, dirs[0], "*")
        test_glob = os.path.join(self.path, dirs[1], "*")

        train_ds = tf.data.Dataset.list_files(train_glob, shuffle=True)
        test_ds = tf.data.Dataset.list_files(test_glob, shuffle=True)

        y_train, y_test = self.load_labels(len(train_ds), len(test_ds))

        self.train = train_ds.map(lambda x : self.parse_image(x, y_train), num_parallel_calls=tf.data.AUTOTUNE)
        self.test = test_ds.map(lambda x : self.parse_image(x, y_test), num_parallel_calls=tf.data.AUTOTUNE)

    def get_split(self, test_ratio):
        if test_ratio is None:
            return self.prepare(self.train, self.test)
        else:
            return super().get_split(test_ratio)
        
    def get_data(self):
        return self.test.concatenate(self.train)

    def parse_label(self, filename, labels):
        file = tf.strings.split(filename, os.sep)[-1]
        index = tf.strings.to_number(tf.strings.split(file, ".")[0], out_type=tf.int32)
        label = labels[index - 1]
        return label

    def load_labels(self, len_train, len_test):
        path = os.path.join(self.path, "list", "train_age.txt")
        train_ds = tf.data.TextLineDataset(path).batch(len_train)
        train = tf.strings.to_number(train_ds.get_single_element())

        path = os.path.join(self.path, "list", "test_age.txt")
        test_ds = tf.data.TextLineDataset(path).batch(len_test)
        test = tf.strings.to_number(test_ds.get_single_element())
        return train, test
    
    def parse_image(self, filename, labels):
        label = self.parse_label(filename, labels)

        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=self.channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.size, self.size])
        return image, label


class FGNetDataset(ImageDataset):

    def __init__(self, path, size, channels) -> None:
        self.path = path
        super().__init__(size, channels)
        

    def parse_label(self, filename):
        parts = tf.strings.split(filename, os.sep)
        label = tf.strings.to_number(tf.strings.substr(parts[-1], 4, 2))
        return label
    
    def load(self):
        glob = os.path.join(self.path, "*")
        list_ds = tf.data.Dataset.list_files(glob, shuffle=False)
        images_ds = list_ds.map(lambda x : self.parse_image(x))
        self.data = images_ds
    
    def get_data(self):
        return self.data

def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    plt.show()
    

def main():
    path = os.path.join("data", "FGNET", "images")
    ds = FGNetDataset(path, size=128, channels=3)
    train, test = ds.get_split(0.2)
    print(train, test)
    print(train.cardinality(), test.cardinality())
    for image, label in train.take(1):
        show(image, label)


if __name__ == "__main__":
    main()
