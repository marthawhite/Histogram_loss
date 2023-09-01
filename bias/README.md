# Bias Analysis

This section focuses on analyzing and approximating the bias incurred by transforming targets to the truncated Gaussian histograms. We consider two sources of bias. The truncation bias is due to using a bounded support for the truncated distribution. The discretization bias is due to the mapping of the continuous distribution to finite bins.

The work is based off derivations from [Imani 2019](https://era.library.ualberta.ca/items/90c26ffa-6eff-4ac6-a011-9699d27d91d0/view/347e81b7-8f26-4acb-9960-044c8a2ee7db/Ehsan_Imani.pdf).

## Scripts
 - `simulation.py` - Approximate the bias induced by the Histogram transformation for a given configuration of the bins.
 - `discretization.py` - Compute the simulated discretization bias over a range of $\sigma_w$ and plot the results.
 - `truncation.py` - Compute the simulated bias over a range of $\psi_\sigma$ and plot the results.
 - `curves.py` - Fit a squared-exponential curve to the simulated biases and plot them compared to the original data. Uses data saved by `discretization.py` and `truncation.py`. You may need to configure file paths.