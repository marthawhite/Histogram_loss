import tensorflow as tf
import os
import matplotlib.pyplot as plt

def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = tf.strings.to_number(tf.strings.substr(parts[-1], 4, 2))

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image, label


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

images_ds = list_ds.map(parse_image)
for image, label in images_ds.shuffle(10000).take(2):
    show(image, label)
    print(image)


