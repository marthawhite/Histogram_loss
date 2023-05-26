import tensorflow as tf
import os
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self) -> None:
        self.seed = 1
        self.ds = self.load()

    def load(self):
        pass

    def get_split(self, test_ratio):
        return tf.keras.utils.split_dataset(self.ds, right_size=test_ratio, shuffle=True, seed=self.seed)


class ImageDataset(Dataset):

    def __init__(self, size, channels) -> None:
        self.size = size
        self.channels = channels
        super().__init__()

    def parse_image(self, filename):
        label = self.parse_label(filename)

        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=self.channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.size, self.size])
        return image, label

    def parse_label(self, filename):
        pass


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
        return images_ds
    

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
