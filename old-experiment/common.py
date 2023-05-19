import numpy as np
import pandas as pd
import pickle
import os
import psutil
import gc

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, GaussianNoise

import tensorflow as tf

from scipy.special import erf

np.random.seed(42)
tf.random.set_seed(42)

adam_eps = None
use_amsgrad = False

# Error measures

def rmse_mae(y_true, y_pred):
    return np.array([np.sqrt(mse(y_true, y_pred)), mae(y_true, y_pred)])


# Models

# Neural network trained with squared error
def create_baseline_nn_model(h_l, lr, input_dim, dropout_rate=0.05, loss='mse', output_noise=None, h_l_trainable=True, clipping=None, fixed_h_l_width=None):
    model = Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=(input_dim,)))
    if fixed_h_l_width is None:
        h_l_width = input_dim/2
    else:
        h_l_width = fixed_h_l_width
    for l in range(h_l):
        model.add(Dense(h_l_width, kernel_initializer='lecun_uniform', activation='relu', trainable=h_l_trainable))
    model.add(Dense(1, kernel_initializer='lecun_uniform', activation='linear'))
    if output_noise is not None:
        model.add(GaussianNoise(output_noise))
    if clipping is None:
        model.compile(loss=loss, optimizer=Adam(lr=lr, epsilon=adam_eps, amsgrad=use_amsgrad))
    else:
        model.compile(loss=loss, optimizer=Adam(lr=lr, epsilon=adam_eps, amsgrad=use_amsgrad, clipnorm=clipping))
    return model

# Neural network trained with the Histogram Loss and Gaussian target distribution
def create_main_model(h_l, lr, n_bins, input_dim, dropout_rate=0.05, loss='categorical_crossentropy', h_l_trainable=True, fixed_h_l_width=None):
    model = Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=(input_dim,)))
    if fixed_h_l_width is None:
        h_l_width = input_dim/2
    else:
        h_l_width = fixed_h_l_width
    for l in range(h_l):
        model.add(Dense(h_l_width, kernel_initializer='lecun_uniform', activation='relu', trainable=h_l_trainable))
    model.add(Dense(n_bins, kernel_initializer='lecun_uniform', activation='softmax'))
    model.compile(loss=loss, optimizer=Adam(lr=lr, epsilon=adam_eps, amsgrad=use_amsgrad))
    return model


# Creating the targets for HL.
# Transform_normal returns a vector whose elements show the area under the target distribution's pdf in each bin's range.

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

    gc.collect()

    return y_tv_dist, y_test_dist, centers

# Data
def get_data(dataset):
    # dataset = 'ctscan' or 'yearpred'

    if dataset == 'yearpred':
        dat = pd.read_csv('dat/YearPredictionMSD.txt', header=None)
        y_min = 1922
        y_max = 2011
    else: # CT Scan
        dat = pd.read_csv('slice_localization_data.csv').iloc[:,1:]
        y_min = 0
        y_max = 100

    return dat, y_min, y_max


# Data train/test splits
def get_split(dat, dataset, test_ratio=0.2, seed=42):
    # dataset = 'ctscan' #'yearpred' or 'ctscan'
    process = psutil.Process(os.getpid())

    if dataset == 'yearpred_fixed':
        tv = dat.iloc[:463715,:]
        test = dat.iloc[463715:,:]

        X_tv = tv.iloc[:,1:]
        y_tv = tv.iloc[:,0]
        X_test = test.iloc[:,1:]
        y_test = test.iloc[:,0]

    elif dataset == 'yearpred':
        X_tv, X_test, y_tv, y_test = train_test_split( dat.iloc[:,1:], dat.iloc[:,0], test_size=test_ratio, random_state=seed)
    else: # CT Scan
        X_tv, X_test, y_tv, y_test = train_test_split( dat.iloc[:,:-1], dat.iloc[:,-1], test_size=test_ratio, random_state=seed)

    X_tv = X_tv.values
    X_test = X_test.values
    y_tv = y_tv.values
    y_test = y_test.values

    print(dat.shape)

    print('mem:', process.memory_info().rss)

    sc = StandardScaler()
    X_tv = sc.fit_transform(X_tv)
    X_test = sc.transform(X_test)

    print(X_tv.shape, y_tv.shape, X_test.shape, y_test.shape)

    return X_tv, X_test, y_tv, y_test


