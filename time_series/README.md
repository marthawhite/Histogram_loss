# Time Series

This directory contains code for evaluating the Histogram Loss on time series forecasting problems.

## Datasets
 - [ETTh1](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720)/ETTm2 - Long term forecasting of power load and oil temperature measurements.

The Electricity Transformer Temperature (ETT) datasets were collected by ([Zhou *et al.* 2021](https://arxiv.org/pdf/2012.07436.pdf)). The *h* variants have hourly measurements, and the *m* variants have recordings taken every 15 minutes. We adapted the standard train-val-test split of 12-4-4 months to a 12-4 train-test split to ensure that results are comparable. 

## Base Models
 - Linear
 - DLinear
 - NLinear
 - Transformer
 - LSTM Encoder-Decoder
 - MLP

The Linear, DLinear, and NLinear models are based on the [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) GitHub repository and corresponding paper ([Zeng *et al.* 2022]((https://arxiv.org/pdf/2205.13504.pdf))). The transformer model is from a [Keras code example](https://keras.io/examples/timeseries/timeseries_classification_transformer/) and is based on the following paper ([Vaswani *et al.* 2017]((https://arxiv.org/pdf/1706.03762.pdf))). The LSTM Encoder-Decoder model uses an LSTM layer with encoder and decoder as MLP blocks of fully-connected layers. We also included a version of the LSTM Encoder-Decoder model that makes autoregressive predictions by feeding the input data back into the model. The MLP model expands on the linear models by using a simple multi-layer perceptron with ReLU activations. We included two variants for predicting features independently or using all features to predict each one. 

# Instruction
1. Setup your Python environment using `requirements.txt`
2. Copy `main.py` to the project (outer) directory
3. (optional) Modify model hyperparameters or datasets to run on by modifying lines 53-71 of `main.py`. Discriptions of hyperparameters is available on lines 26-50 of `main.py`
4. Run `main.py <base_model> <loss>` 
    where `base_model` is one of: `transformer`, `LSTM`, `linear`, `independent_dense`, or `dependent_dense`. And `loss` is one of: `HL` or `L2`
5. Collect training progress results in `{loss}_{dataset}_{base_model}.json`

Note that you can replace `main.py` with `model_analysis.py` in the above procedure to get the training progress results as well as the test set targets and model prediction after the last training epoch, as `{dataset}_targets.npy` and `{dataset}_{base_model}_{loss}.npy` respectively.