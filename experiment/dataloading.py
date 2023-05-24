import tensorflow as tf
import os
import matplotlib.pyplot as plt

def parse_label(filename):
    parts = tf.strings.split(filename, os.sep)
    label = tf.strings.to_number(tf.strings.substr(parts[-1], 4, 2))
    return label

def parse_image(filename, size=128, channels=3):
    label = parse_label(filename)

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [size, size])
    return image, label

def parse_wrapper(size, channels):
    return lambda x : parse_image(x, size, channels)


path = os.path.join("data", "FGNET", "images", "*")
list_ds = tf.data.Dataset.list_files(path, shuffle=False)
for item in list_ds.take(2):
    print(item)


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    plt.show()


img_size = 128
img_channels = 3

images_ds = list_ds.map(parse_wrapper(img_size, img_channels))
for image, label in images_ds.shuffle(10000).take(2):
    show(image, label)
    print(image)


