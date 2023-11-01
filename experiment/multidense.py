"""Class for multidimensional dense layers."""

import tensorflow as tf
from tensorflow import keras


class MultiDense(keras.layers.Layer):
    """Multidimensional dense layer.

    Maps the last dimension of input to a multidimensional output. 
    Creates an array of dense tensors that are applied across the other input dimensions.
    
    For an input shape (n, x1, ..., xd) with shape (y1, ..., yk), this layer outputs
    (n, x1, ..., x(d-1), y1, ..., yk) through multiplication with a (x1, ..., x(d-1)) shaped collection of 
    (xd, y1, ..., yk) tensors.

    Params:
        shape - the shape added to the output layer; replaces the last dimension of the input
        individual - True if the multiplication should use a different tensor for each input dimension
            if False, the same tensor is applied across all input dimensions
    """

    def __init__(self, shape, individual=True):
        super().__init__()
        self.shape = shape
        self.d = len(self.shape)
        self.individual = individual

        # Use matvec for 1d output
        # TODO: Implement using (sub)classes
        if self.d == 0:
            self.f = tf.linalg.matvec
        else:
            self.f = tf.matmul

    def build(self, input_shape):
        """Create the kernel for a specified input shape.
        
        Params:
            input_shape - the shape of the batches passed to the layer (includes batchsize)
        """
        
        # Initialize the bias and matrix shape
        if self.individual:
            mat_shape = self.shape[:-1] + input_shape[1:] + self.shape[-1:]
            bias_shape = input_shape[1:-1] + self.shape
        else:
            mat_shape = self.shape[:-1] + tuple([1 for x in input_shape[1:-1]]) + input_shape[-1:] + self.shape[-1:]
            bias_shape = self.shape

        # Create kernel
        self.w = self.add_weight(
            name="w",
            shape= mat_shape,
            initializer="lecun_normal", 
            trainable=True
        )
        self.b = self.add_weight(
            name="b",
            initializer="lecun_normal",
            shape=bias_shape,
            trainable=True
        )
        
        # Specify permutations
        k = len(input_shape) - 1
        self.input_perm = list(range(1, k)) + [0, k]
        l = k + self.d

        # Hacky workaround for 1D output
        # Not guaranteed to work for > 1 input dimensions
        # TODO: FIX
        if self.d == 0:
            l -= 1
        if l > 0:
            self.output_perm = [(l-2) % (l+1)] + list(range(self.d-1, l - 2)) + list(range(0, self.d-1)) + [l-1]
        else:
            self.output_perm = [0]   


    def call(self, inputs):
        """Apply the dense layers to the inputs.
        
        Params:
            inputs - the batched inputs to transform with shape (n, x1, ..., xd)

        Returns:
            the output tensor of shape (n, x1, ..., x(d-1), y1, ..., yk)
        """
        inputs = tf.transpose(inputs, self.input_perm)
        outputs = self.f(inputs, self.w)
        return tf.transpose(outputs, self.output_perm) + self.b
