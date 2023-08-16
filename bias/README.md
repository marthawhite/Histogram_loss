# Bias Analysis

This section focuses on analyzing and approximating the bias incurred by transforming targets to the truncated Gaussian histograms. We consider two sources of bias. The truncation bias is due to using a bounded support for the truncated distribution. The discretization bias is due to the mapping of the continuous distribution to finite bins.

The work is based off derivations from [Imani 2019](https://era.library.ualberta.ca/items/90c26ffa-6eff-4ac6-a011-9699d27d91d0/view/347e81b7-8f26-4acb-9960-044c8a2ee7db/Ehsan_Imani.pdf).