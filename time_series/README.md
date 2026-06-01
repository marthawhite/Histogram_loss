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
1. Set up your Python 3.10 environment using `requirements.txt`
2. Copy `main.py` to the project (outer) directory
3. (optional) Override any hyperparameter on the command line — every field of the `Config` dataclass in `main.py` is a flag. Run `python main.py --help` to list them.
4. Run `python main.py --base-model <base_model> --loss <loss> [--seed <seed>]`
    where `base_model` is one of: `transformer`, `transformer_large`, `transformer_enc_dec`, `autoformer`, `LSTM`, `GRU`, `linear`, `independent_dense`, or `dependent_dense`; `loss` is `HL` or `L2`; and `seed` is an optional integer (default 1). Vary `--seed` for multi-seed runs, e.g. `python main.py --base-model LSTM --loss HL --seed 3`.

Note that you can replace `main.py` with `model_analysis.py` in the above procedure to get the training progress results as well as the test set targets and model prediction after the last training epoch, as `{dataset}_targets.npy` and `{dataset}_{base_model}_{loss}.npy` respectively.