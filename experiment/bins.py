"""Module for producing and working with histogram bins."""

import tensorflow as tf


def get_bins(n_bins, pad_ratio, sig_ratio, low=0., high=1.):
    """Return the histogram bins given the HL parameters.
    Produces n_bins bins with pad_ratio * sig_ratio * bin_width padding on each side. 
    
    Produces bins with width according to the following equation:
        w = (high - low) / (n_bins - 2 * pad_ratio * sig_ratio)

    Note: The low/high params must be broadcastable to the target shape to use the 
    bins directly for HL-Gaussian. If the dimensions are a subset of the target shape,
    you must add the appropriate axes with tf.expand_dims before using.

    Params:
        n_bins - the number of bins to create (includes padding)
        pad_ratio - the number of sigma of padding to use on each side 
        sig_ratio - the ratio of sigma to bin width
        low - the lower bounds of the histogram support; shape (x1, ..., xn)
        high - the upper boudns of the histogram support; same shape as low

    Returns: 
        borders - a Tensor of shape (n_bins + 1, x1, ..., xn) of bin borders
        sigma - the sigma to use for HL-Gaussian of shape (x1, ..., xn)
    """
    bin_width = (high - low) / (n_bins - 2 * sig_ratio * pad_ratio)
    pad_width = sig_ratio * pad_ratio * bin_width
    borders = tf.linspace(low - pad_width, high + pad_width, n_bins + 1)
    sigma = bin_width * sig_ratio
    return borders, sigma
