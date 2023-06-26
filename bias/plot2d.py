import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def get_difs(inputs, sigma):
    

def transform(inputs, borders, sigma):
    border_targets = adjust_and_erf(borders, np.expand_dims(inputs, -1), sigma)
    two_z = border_targets[:, -1] - border_targets[:, 0]
    x_trans = (border_targets[:, 1:] - border_targets[:, :-1]) / np.expand_dims(two_z, -1)
    return x_trans


def adjust_and_erf(a, mu, sig):
    return erf((a - mu) / (np.sqrt(2.0) * sig))

steps = 100
k = np.linspace(-0.5, 0.5, steps + 1)
sig_ratio = np.linspace(0, 5, steps + 1)[1:]

outputs = 