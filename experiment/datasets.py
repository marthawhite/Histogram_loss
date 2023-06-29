import tensorflow as tf
import os
import pandas as pd

class Dataset:
    """Base dataset class."""

    def __init__(self, buffer_size=None, batch_size=32, prefetch=1) -> None:
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.buf = buffer_size
        self.load()

    def prepare(self, splits):
        data = []
        for x in splits:
            x = self.shuffle(x)
            x = x.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            x = x.batch(self.batch_size).prefetch(self.prefetch)
            data.append(x)
        return data

    def load(self):
        pass

    def shuffle(self, data, reshuffle=True):
        if self.buf is None:
            buf = len(self)
        else:
            buf = self.buf
        return data.shuffle(buf, reshuffle_each_iteration=reshuffle)

    def preprocess(self, *args):
        return args

    def get_data(self):
        pass

    def __len__(self):
        return len(self.get_data())

    def get_split(self, val_ratio, test_ratio=None, shuffle=False):
        data = self.get_data()
        
        if shuffle:
            data = self.shuffle(data, False)

        splits = self.split(data, val_ratio, test_ratio)        

        return self.prepare(splits)
    
    def split(self, data, val_ratio, test_ratio):
        if test_ratio is not None:
            return self.three_split(data, val_ratio, test_ratio)
        else:
            return self.two_split(data, val_ratio)
    
    def two_split(self, data, test_ratio):
        test_len = self.get_num(test_ratio)
        train_len = len(self) - test_len

        train = data.take(train_len)
        test = data.skip(train_len).take(test_len)
        return train, test

    def get_num(self, ratio):
        if ratio >= 1:
            return ratio
        else:
            return int(len(self) * ratio)

    def three_split(self, data, val_ratio, test_ratio):
        test_len = self.get_num(test_ratio)
        val_len = self.get_num(val_ratio)
        train_len = len(data) - val_len - test_len

        train = data.take(train_len)
        val = data.skip(train_len).take(val_len)
        test = data.skip(train_len + val_len)
        return train, val, test


class CSVDataset(Dataset):

    def __init__(self, path, targets, drop_cols=[], **kwargs) -> None:
        self.path = path
        self.targets = targets
        self.drop_cols = drop_cols
        super().__init__(**kwargs)

    def load(self):
        df = pd.read_csv(self.path)
        df = df.drop(self.drop_cols, axis=1)
        x = df.drop(self.targets, axis=1)
        y = df[self.targets]
        ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x, dtype=tf.float32),tf.convert_to_tensor(y, dtype=tf.float32)))
        self.ds = ds

    def get_data(self):
        return self.ds
    

class TSDataset(Dataset):

    def __init__(self, path, seq_len, pred_len, targets=None, drop_cols=[], mode='M', overlap=0, **kwargs) -> None:
        self.path = path
        self.targets = targets
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.overlap = overlap
        self.drop_cols = drop_cols
        super().__init__(**kwargs)

    def load(self):
        df = pd.read_csv(self.path)
        df = df.drop(self.drop_cols, axis=1)

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
        return self.ds
    
    def __len__(self):
        return self.n


class ImageDataset(Dataset):

    def __init__(self, size=128, channels=3, **kwargs) -> None:
        self.size = size
        self.channels = channels
        super().__init__(**kwargs)

    def preprocess(self, x):
        return self.parse_image(x)

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
    
    def __init__(self, path, aligned=True, **kwargs) -> None:
        self.path = path
        self.aligned = aligned
        super().__init__(**kwargs)

    def load(self):
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
        if val_ratio is None:
            return self.train, self.test
        else:
            return super().split(data, val_ratio, test_ratio)
        
    def get_data(self):
        return self.train.concatenate(self.test)

    def parse_label(self, filename):
        path = tf.strings.split(filename, os.sep)
        
        img_dir = path[-2]
        file = path[-1]
        index = tf.strings.to_number(tf.strings.split(file, ".")[0], out_type=tf.int32)
        index = tf.where(tf.equal(img_dir, "test"), index + self.n_train, index)
        return self.labels[index - 1]

    def load_labels(self, len_train, len_test):
        path = os.path.join(self.path, "list", "train_age.txt")
        train_ds = tf.data.TextLineDataset(path).batch(len_train)
        train = tf.strings.to_number(train_ds.get_single_element())

        path = os.path.join(self.path, "list", "test_age.txt")
        test_ds = tf.data.TextLineDataset(path).batch(len_test)
        test = tf.strings.to_number(test_ds.get_single_element())
        return train, test

class FGNetDataset(ImageDataset):

    def __init__(self, path, **kwargs) -> None:
        self.path = path
        super().__init__(**kwargs)
        
    def parse_label(self, filename):
        parts = tf.strings.split(filename, os.sep)
        label = tf.strings.to_number(tf.strings.substr(parts[-1], 4, 2))
        return label
    
    def load(self):
        glob = os.path.join(self.path, "*")
        list_ds = tf.data.Dataset.list_files(glob, shuffle=False)
        self.data = list_ds
    
    def get_data(self):
        return self.data
    

class UTKFaceDataset(ImageDataset):

    def __init__(self, path, **kwargs):
        self.path = path
        super().__init__(**kwargs)
        
    def parse_label(self, filename):
        parts = tf.strings.split(filename, os.sep)
        label = tf.strings.to_number(tf.strings.split(parts[-1], "_")[0])
        return label
    
    def load(self):
        glob = os.path.join(self.path, "*")
        list_ds = tf.data.Dataset.list_files(glob, shuffle=False)
        self.data = list_ds

    def get_data(self):
        return self.data
    