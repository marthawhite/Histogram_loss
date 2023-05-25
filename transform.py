import tensorflow as tf
import numpy as np
from scipy.special import erf
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


def adjust_and_erf(a, mu, sig):
    return erf((a - mu)/(np.sqrt(2.0)*sig))

def transform_normal(y_tv, y_test, y_min, y_max, n_bins=100, ker_par_ratio=1.0):
    '''
    n_bins: Number of centers
    ker_par_ratio: The ratio between sig and bin size
    '''
    # Creating new labels
    eps = 1e-7
    bin_size = (y_max + eps - y_min)*1.0/n_bins
    ker_par = bin_size * ker_par_ratio # Sigma for Gaussian

    borders = np.linspace(y_min, y_max+eps, n_bins+1)
    centers = borders[:-1] + bin_size/2.0

    # Distribution
    border_targets_tv = adjust_and_erf(borders[np.newaxis,:], y_tv[:,np.newaxis], ker_par)
    two_z_tv = border_targets_tv[:,-1] - border_targets_tv[:,0]
    y_tv_dist = (border_targets_tv[:,1:] - border_targets_tv[:,:-1])/two_z_tv[:,np.newaxis]

    border_targets_test = adjust_and_erf(borders[np.newaxis,:], y_test[:,np.newaxis], ker_par)
    two_z_test = border_targets_test[:,-1] - border_targets_test[:,0]
    y_test_dist = (border_targets_test[:,1:] - border_targets_test[:,:-1])/two_z_test[:,np.newaxis]


    return y_tv_dist, y_test_dist, centers

def main():
    y_train = tf.random.uniform((100,), 0, 100)
    y_test = tf.random.uniform((10,), 0, 100)

    y_min = 0
    y_max = 100

    train, test, centers = transform_normal(y_train, y_test, y_min, y_max)

    print(train.shape, test.shape, centers.shape)
    y_recreated = tf.convert_to_tensor(np.dot(train, centers), dtype=tf.float32)
    print(tf.math.reduce_mean(abs(y_train - y_recreated)))
    


main()
