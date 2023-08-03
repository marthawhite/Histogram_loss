import tensorflow as tf


def get_bins(n_bins, pad_ratio, sig_ratio, low=0., high=1.):
    """Return the histogram bins given the HL parameters.
    
    Params:
        n_bins - the number of bins to create (includes padding)
        pad_ratio - the number of sigma of padding to use on each side 
        sig_ratio - the ratio of sigma to bin width

    Returns: 
        borders - a Tensor of n_bins + 1 bin borders
        sigma - the sigma to use for HL-Gaussian
    """
    bin_width = (high - low) / (n_bins - 2 * sig_ratio * pad_ratio)
    pad_width = sig_ratio * pad_ratio * bin_width
    borders = tf.linspace(low - pad_width, high + pad_width, n_bins + 1)
    sigma = bin_width * sig_ratio
    return borders, sigma
