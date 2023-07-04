import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class Scaler:
    """Apply min-max scaling to the target of each sample.
    
    Params:
        y_min - preset minimum value
        y_max - preset maximum value
    """

    def __init__(self, y_min=1e9, y_max=-1e9) -> None:
        self.y_min = y_min
        self.y_max = y_max

    def fit(self, data):
        """Update the minimum and maximum values according to the data.
        
        Params:
            data - a tf.data.Dataset of (x, y) tuples of sample batches
        """
        for x, y_batch in data:
            for y in y_batch:
                self.y_min = tf.where(y < self.y_min, y, self.y_min)
                self.y_max = tf.where(y > self.y_max, y, self.y_max)

    def transform(self, data):
        """Transform the data using min-max scaling on the target of each sample.
        
        Params:
            data - a tf.data.Dataset of (x, y) tuples of sample batches

        Returns: a tf.data.Dataset where the targets have been scaled
        """
        y_range = self.y_max - self.y_min
        scale = tf.where(y_range == 0, tf.ones_like(y_range), y_range)
        return data.map(lambda x, y: (x, (y - self.y_min) / scale))
    

class Normalizer:
    """Apply normalization to the input data of each sample."""

    def __init__(self) -> None:
        self.sc = StandardScaler()

    def fit(self, data):
        """Determine the mean and standard deviation by column from the data.
        
        Params:
            data - a tf.data.Dataset of (x, y) tuples of sample batches
        """
        for x_batch, y in data:
            self.sc.partial_fit(x_batch.numpy())

    def transform(self, data):
        """Transform the data by normalizing the input for each sample.
        
        Params:
            data - a tf.data.Dataset of (x, y) tuples of sample batches

        Returns: a tf.data.Dataset where the inputs have been normalized
        """
        mu = tf.cast(self.sc.mean_, tf.float32)
        std = tf.cast(tf.math.sqrt(self.sc.var_), tf.float32)
        scale = tf.where(std == 0., tf.ones_like(std), std)
        return data.map(lambda x, y: ((x - mu) / scale, y))
