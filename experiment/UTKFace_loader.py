import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
from experiment.datasets import ImageDataset, Dataset
import numpy as np

class UTKFaceDataset(ImageDataset):
    def __init__(self, path, size, channels):
        self.path = path
        super().__init__(size, channels)
        
    def parse_label(self, filename):
        parts = tf.strings.split(filename, os.sep)
        label = tf.strings.to_number(tf.strings.split(parts[-1], "_")[0])
        return label
    
    def load(self):
        glob = os.path.join(self.path, "*")
        list_ds = tf.data.Dataset.list_files(glob, shuffle=False)
        images_ds = list_ds.map(lambda x : self.parse_image(x))
        self.data = images_ds

    def get_data(self):
        return self.data
    
