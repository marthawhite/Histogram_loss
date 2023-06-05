"""Keras implementations of models from models.py

Includes HL Gaussian, HL OneBin, and Regression models.
"""


from tensorflow import keras
import tensorflow as tf


class TruncGaussHistTransform(keras.layers.Layer):
    """Layer that transforms a scalar target into a binned probability vector 
    that approximates a truncated Gaussian distribution with the target as the mean.
    
    Params:
        borders - the borders of the histogram bins
        sigma - the sigma parameter of the truncated Gaussian distribution
    """

    def __init__(self, borders, sigma):
        super().__init__(trainable=False, name="TruncGaussHistTransform")
        self.borders = borders
        self.sigma = sigma

    def call(self, inputs):
        """Transform the input and return it.
        
        Params:
            inputs - the tensor of targets to transform

        Returns:
            x_transformed - a tensor of shape (len(inputs), len(borders) - 1)
            consisting of the probability vectors for each target
        """
        border_targets = self.adjust_and_erf(self.borders, tf.expand_dims(inputs, 1), self.sigma)
        two_z = border_targets[:, -1] - border_targets[:, 0]
        x_transformed = (border_targets[:, 1:] - border_targets[:, :-1]) / tf.expand_dims(two_z, 1)
        return x_transformed

    def adjust_and_erf(self, a, mu, sig):
        """Calculate the erf of a after standardizing and dividing by sqrt(2)."""
        return tf.math.erf((a - mu)/(tf.math.sqrt(2.0)*sig))
    

class OneHotTransform(keras.layers.Layer):
    """Layer that transforms a target into a one-hot representation
    based on the histogram bin that it lies in.
    
    Params:
        borders - the borders of the histogram bins
    """

    def __init__(self, borders):
        super().__init__(trainable=False, name="OneHotTransform")
        self.borders = borders
        self.bin_size = borders[1] - borders[0]
        self.low = tf.reduce_min(borders)
        self.n_classes = tf.size(borders) - 1

    def call(self, inputs):
        """Transform the input and return it.
        
        Params:
            inputs - the tensor of targets to transform

        Returns:
            a tensor of shape (len(inputs), len(borders) - 1)
            consisting of the one-hot vectors for each target
        """
        adjusted = (inputs - self.low) / self.bin_size
        indices = tf.cast(adjusted, tf.int32)
        return tf.one_hot(indices, self.n_classes, dtype=tf.float32)
    

class HistMean(keras.layers.Layer):
    """Layer that transforms a probability vector into its expected value.
    
    Params:
        centers - the centers of the histogram bins
    """

    def __init__(self, centers):
        super().__init__(trainable=False, name="HistMean")
        self.centers = centers

    def call(self, inputs):
        """Return the weighted average between the centers and probability vectors.
        
        Params:
            inputs - a tensor of probability vectors to transform

        Returns:
            a tensor of shape (len(inputs), 1) consisting of the expected values
        """
        return tf.linalg.matvec(inputs, self.centers)


class Regression(keras.Model):
    """Model that performs regression using features from a given backbone model.
    
    Params:
        base - the backbone model to learn features
    """

    def __init__(self, base, dropout=0):
        super().__init__()
        self.base = base
        self.dropout = keras.layers.Dropout(dropout)
        self.reg = keras.layers.Dense(1)

    def call(self, inputs, training=None):
        """Perform regression on the features outputted by the base model.
        
        Params:
            inputs - the tensor to give to the base model
        
        Returns:
            a tensor with shape (len(inputs), 1) consisting of the regression predictions
            for the given inputs
        """
        features = self.base(inputs, training=training)
        features = self.dropout(features, training=training)
        return self.reg(features)


class HistModel(keras.Model):
    """Model that performs regression on features from a given backbone model 
    using a histogram loss.

    Params:
        base - the backbone model to learn features
        centers - the centers of the histogram bins
        transform - the histogram transform to apply to the targets
            to facilitate learning
        name - the name of the model
    """

    def __init__(self, base, centers, transform, name="HistModel", dropout=0):
        super().__init__(name=name)
        self.base = base
        self.dropout = keras.layers.Dropout(dropout)
        self.softmax = keras.layers.Dense(tf.size(centers), activation="softmax")
        self.transform = transform
        self.mean = HistMean(centers)

    def call(self, inputs, training=None):
        """Perform regression on the inputs using the histogram loss model.
        
        Params:
            inputs - the tensor to give to the base model
            training - flag indicating whether the model is called during training
        
        Returns:
            a tensor of shape (len(inputs), len(centers)) consisting of the expected 
            values of the binned probability vectors obtained from the inputs
        """
        features = self.base(inputs, training)
        features = self.dropout(features, training=training)
        hist = self.softmax(features, training=training)
        return self.mean(hist)

    def train_step(self, data):
        """Update the model weights and metrics based on a single batch of data.
        
        Params:
            data - a batch of data in the form (x, y)
                typically a tf.data.Dataset where elements are a tuple of tensors

        Returns: a dict containing the metric values computed on data
        """
        x, y = data
        y_transformed = self.transform(y)

        with tf.GradientTape() as tape:
            features = self.base(x, training=True)
            features = self.dropout(features, training=True)
            hist = self.softmax(features, training=True)
            loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
    

class HLGaussian(HistModel):
    """Keras model using a histogram loss with a truncated Gaussian 
    distribution on the targets.
    
    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        sigma - the sigma parameter of the truncated Gaussian distribution
    """

    def __init__(self, base, borders, sigma, dropout):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = TruncGaussHistTransform(borders, sigma)
        super().__init__(base, centers, transform, "HL-Gaussian", dropout)


class HLOneBin(HistModel):
    """Keras model using a histogram loss with one-hot encoding of the targets.
    
    Params:
        base - the backbone model used to learn features
        borders - the borders of the histogram bins
        sigma - the sigma parameter of the truncated Gaussian distribution
    """

    def __init__(self, base, borders, dropout):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = OneHotTransform(borders)
        super().__init__(base, centers, transform, "HL-OneBin", dropout)
    

def main():
    base = keras.layers.Dense(100)
    borders = tf.linspace(0, 100, 100)
    hl = HLGaussian(base, borders, 1.)
    hl.compile()
    print(hl.summary())




if __name__ == "__main__":
    main()
