import tensorflow as tf

class HistNormTransform:
    """Transforms each datapoint into a vector of histogram probabilities
    approximating a truncated Gaussian distribution whose mean is the datapoint.
    """
    
    def __init__(self, n_bins=100, sig_ratio=1.0, padding=0.1) -> None:
        """Initialize the transform.
        
        Params:
            self - the transform object
            n_bins - the number of equal-width bins for the histogram
            sig_ratio - the sigma parameter of the truncated Gaussian as a fraction
                of the bin width
            padding - the amount of extra support outside of the range of data
                if in [0, 1), the fraction of the range to add to each side
                if > 1, the number of bins to add on each side
        """

        self.n_bins = n_bins
        self.sig_ratio = sig_ratio
        self.pad_ratio = padding

    @classmethod
    def from_bins(cls, borders, sigma):
        """Create a histogram transform from preset bins.
        Note: This transform should not be fitted!

        Params:
            cls - the HistNormTransform class
            borders - the borders of the histogram bins
            sigma - the sigma parameter of the truncated Gaussian
        """

        obj = cls.__new__(cls)
        super(HistNormTransform, obj).__init__()
        obj.borders = borders
        obj.sigma = sigma
        obj.centers = (obj.borders[1:] + obj.borders[:-1]) / 2.0
        obj.n_bins = obj.centers.shape[0]
        return obj

    def get_centers(self):
        """Return the centers of the histogram bins."""
        return self.centers

    def get_min_max(self, x):
        """Return the bounds of the histogram support after applying padding."""

        # Adjust padding if expressed as number of bins
        if self.pad_ratio >= 1:
            bins = self.n_bins
            self.n_bins += 2 * self.pad_ratio
            self.pad_ratio = self.pad_ratio * 1.0 / bins
        
        # Add padding
        x_max = tf.math.reduce_max(x)
        x_min = tf.math.reduce_min(x)
        padding = (x_max - x_min) * self.pad_ratio        
        x_max += padding
        x_min -= padding

        return x_min, x_max

    def fit(self, x):
        """Create the histogram bins based on the min and max from training data.
        
        Params:
            self - the transform object
            x - the training data to learn the bins on
        """
        x_min, x_max = self.get_min_max(x)
        bin_size = (x_max - x_min) / self.n_bins
        self.sigma = self.sig_ratio * bin_size
        self.borders = tf.linspace(x_min, x_max, self.n_bins + 1)
        self.centers = self.borders[:-1] + bin_size / 2.0

    def transform(self, x):
        """Transform data into binned probability vectors.
        
        Params:
            self - the transform object
            x - the data to transform

        Returns: array of shape (x.shape, n_bins)
        """

        border_targets = self.adjust_and_erf(self.borders, tf.expand_dims(x, 1), self.sigma)
        two_z = border_targets[:,-1] - border_targets[:,0]
        x_transformed = (border_targets[:, 1:] - border_targets[:, :-1]) / tf.expand_dims(two_z, 1)
        return x_transformed

    def fit_transform(self, x):
        """Fit and transform learning data into probability vectors.
        
        Params:
            self - the transform object
            x - the data to train and transform with

        Returns: array of shape (x.shape, n_bins)
        """

        self.fit(x)
        return self.transform(x)
    
    def __str__(self) -> str:
        return f"n_bins: {self.n_bins}, Borders: {self.borders}"

    def adjust_and_erf(self, a, mu, sig):
        """Compute the complex error function after standardizing."""
        return tf.math.erf((a - mu)/(tf.math.sqrt(2.0)*sig))


def main():
    

    y_train = tf.range(0, 101, 1, dtype=tf.float32)

    ht = HistNormTransform(100, 2, 0)
    yt_train = ht.fit_transform(y_train)

    # Assert that all outputs are probability distributions
    eps = 1e-3
    sums = tf.math.reduce_sum(yt_train, axis=1)
    assert tf.math.reduce_all(1 - eps < sums) and tf.math.reduce_all(sums < 1 + eps)

    # Check the mean absolute error when autoencoding
    centers = ht.get_centers()
    y_recreated = tf.linalg.matvec(yt_train, centers)
    abs_err = tf.math.abs(y_train - y_recreated)
    print(tf.math.reduce_mean(abs_err))

    print(ht)

    ht2 = HistNormTransform.from_bins(y_train, 1)
    print(ht2)

if __name__ == "__main__":
    main()
