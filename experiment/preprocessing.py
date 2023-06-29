import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class Scaler:

    def __init__(self, y_min=1e9, y_max=-1e9) -> None:
        self.y_min = y_min
        self.y_max = y_max

    def fit(self, data):
        for x, y_batch in data:
            for y in y_batch:
                self.y_min = tf.where(y < self.y_min, y, self.y_min)
                self.y_max = tf.where(y > self.y_max, y, self.y_max)

    def transform(self, data):
        y_range = self.y_max - self.y_min
        scale = tf.where(y_range == 0, tf.ones_like(y_range), y_range)
        return data.map(lambda x, y: (x, (y - self.y_min) / scale))
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    

class Normalizer:

    def __init__(self) -> None:
        self.sc = StandardScaler()

    def fit(self, data):
        for x_batch, y in data:
            self.sc.partial_fit(x_batch.numpy())

    def transform(self, data):
        mu = tf.cast(self.sc.mean_, tf.float32)
        std = tf.cast(tf.math.sqrt(self.sc.var_), tf.float32)
        scale = tf.where(std == 0., tf.ones_like(std), std)
        return data.map(lambda x, y: ((x - mu) / scale, y))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
