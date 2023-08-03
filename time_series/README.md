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

The Linear, DLinear, and NLinear models are based on the [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) GitHub repository and corresponding paper ([Zeng *et al.* 2022]((https://arxiv.org/pdf/2205.13504.pdf))). The transformer model is from a [Keras code example](https://keras.io/examples/timeseries/timeseries_classification_transformer/) and is based on the following paper ([Vaswani *et al.* 2017]((https://arxiv.org/pdf/1706.03762.pdf))). The LSTM Encoder-Decoder model uses an LSTM layer with encoder and decoder as MLP blocks of fully-connected layers. We also included a version of the LSTM Encoder-Decoder model that makes autoregressive predictions by feeding the input data back into the model.