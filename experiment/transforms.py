"""
Module containing histogram transforms for targets.
"""

import tensorflow as tf
from tensorflow import keras


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
    

class UniformTransform(keras.layers.Layer):
    """Transform the target using a mixture of Dirac delta and uniform distributions.
    A target y is mapped to a binned probability vector with values epsilon if 
    y is not in the bin and 1 - (k-1) * epsilon if y is in the bin.

    Params:
        borders - the borders of the histogram bins
        eps - the uniform noise parameter
    """

    def __init__(self, borders, eps):
        super().__init__(trainable=False, name="UniformTransform")
        self.onehot = OneHotTransform(borders)
        self.eps = eps
        k = tf.size(borders) - 1
        self.scale = 1 - k * self.eps

    def call(self, inputs):
        """Transform the input and return it.
        
        Params:
            inputs - the tensor of targets to transform

        Returns: 
            a tensor of shape (len(inputs), len(borders) - 1) consisting of the
            binned probability vectors
        """
        onehot = self.onehot(inputs)
        return onehot * self.scale + self.eps
    

class ProjTransform(keras.layers.Layer):
    """Project the target uniformly onto the two nearest histogram bins.
    
    Params:
        centers - the bins centers
    """

    def __init__(self, centers):
        super().__init__(trainable=False, name="ProjTransform")
        self.w = centers[1] - centers[0]
        self.low = centers[0]
        self.centers = centers

    def call(self, inputs):
        """Return the binned probability vectors for the inputs.
        
        Params:
            inputs - the targets to transform

        Returns: a tensor of shape (len(inputs), len(centers)) containing the probability
            of the target falling in each bin
        """
        i = tf.cast(tf.math.floordiv(inputs - self.low, self.w), tf.int32)
        m = tf.gather_nd(self.centers, tf.expand_dims(i, 1))
        p = (inputs - m) / self.w
        n = tf.size(inputs)
        inds = tf.range(0, n)
        indices = tf.concat([tf.stack([inds, i], 1), tf.stack([inds, i+1], 1)], 0)
        values = tf.concat([1 - p, p], 0)
        return tf.scatter_nd(indices, values, (n, tf.size(self.centers)))


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