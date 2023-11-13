"""Module containing the run code for time series experiments.

Usage: 
    python main.py data_path

Params:
    data_path - path to the input data file
"""

import tensorflow as tf
from tensorflow import keras
from experiment.models import HLGaussian, Regression
from time_series.base_models import transformer, linear, lstm_encdec
import json
from experiment.bins import get_bins
from time_series.datasets import get_time_series_dataset
import wandb

def training(model,train,test,epochs,optimizer,pred_len,loss):
    mse_test_metric = keras.metrics.MeanSquaredError(name="mse")
    mae_test_metric = keras.metrics.MeanAbsoluteError(name="mae")
    wandb.define_metric("custom_step")
    wandb.define_metric("mse_test_loss", step_metric="custom_step")
    wandb.define_metric("mae_test_loss", step_metric="custom_step")
    for epoch in range(epochs):
        train.shuffle(len(train))
        for step, (x_batch_train, y_batch_train) in enumerate(train):
            with tf.GradientTape() as tape:
                preds = model(x_batch_train, training=True)  
                loss_value = loss(y_batch_train, preds)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step%100:
                wandb.log({"training_loss":loss_value.numpy().item()})
        for test_step,(x_batch_val, y_batch_val) in enumerate(test):
            test_pred = model(x_batch_val, training=False)
            mse_test_metric.update_state(y_batch_val, test_pred)
            mae_test_metric.update_state(y_batch_val, test_pred)   
        test_mse_acc = mse_test_metric.result()
        test_mae_acc = mae_test_metric.result()
        mse_test_metric.reset_states()
        mae_test_metric.reset_states()
        res = {'mse_test_loss':test_mse_acc.numpy().item(), 'mae_test_loss':test_mae_acc.numpy().item(),"custom_step":epoch}
        wandb.log(res)
    wandb.run.summary["mse_test_loss"] = test_mse_acc.numpy().item()
    wandb.run.summary["mae_test_loss"] = test_mae_acc.numpy().item()
    ### Log the predictions on one batch of the test data
    for test_step,(x_batch_val, y_batch_val) in enumerate(test):
            test_pred = model(x_batch_val, training=False)
            for i in range(len(test_pred)):
                wandb.log({"test_prediction":test_pred[i].numpy().item(),"test_target":y_batch_val[i].numpy().item()})
            break
    return
    
def main(base_model, loss):
    """Run the time series experiment.
    
    Params:
        base_model(str):  Name of the base model
        loss(str): Name of loss, either HL or L2
        
    General Params:
        datasets: Array of the names of the datasets to train and test the model on
        seq_len: Number of previous timesteps to be given as input to the model (this is the same for all channels)
        pred_len: Number of future timesteps the model will predict (this is the same for all channels)
        epochs: Number of epochs to train for
        sig_ratio: Sigma ratio of the discretized histogram transform
        pad_ratio: Padding ratio of the discretized histogram transform
        n_bins: Number of bins in the discretized histogram
        chans: Number of variables in the dataset
        test_ratio: Ratio of test data to train data
        batch_size: Batch Size for the training
        drop: Name of column(s) that contain data that should not be used as inputs (for multiple columns use a list of strings)
        metrics: Metrics for evaluating train and test performance
        lr: Learning rate
        input_target_offset: Number of steps between the last input time step and the first target time step
        
    Model Specific Params:
        Transformer:
            head_size: Size of head for key and query
            n_heads: Number of attention heads
            features: Number of feature dimensions
        LSTM:
            layers: Number of layers in the encoder and decoder
            width: Width of the layers in the encoder and decoder
        
        
    """
    
    configs = {
    "datasets" : ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    "pred_len" : 1,
    "seq_len" : 336,
    "epochs" : 25,
    "sig_ratio" : 2.,
    "pad_ratio" : 3.,
    "n_bins" : 100,
    "chans" : 1, # the number of target prediction variables
    "input_channels":7,
    "head_size" : 256,
    "n_heads" : 3,
    "features" : 64,
    "layers" : 1,
    "width" : 256,
    "test_ratio" : 0.25,
    "batch_size" : 32,
    "drop" : "date",
    "lr" : 0.0001,
    "input_target_offset" : 96,
    "base_model":base_model,
    "loss":loss,
    "univariate":True, ## code is only doing univariate for now
    }
    for dataset in configs["datasets"]:
        configs["dataset"] = dataset
        keras.utils.set_random_seed(1)
        data_path = f"{dataset}.csv"
        train, test, dmin, dmax = get_time_series_dataset(data_path, configs["drop"], configs["seq_len"], configs["batch_size"], configs["chans"], configs["input_target_offset"],configs["univariate"])

        borders, sigma = get_bins(configs["n_bins"], configs["pad_ratio"], configs["sig_ratio"], dmin, dmax)
        borders = tf.expand_dims(borders, -1)
        sigma = tf.expand_dims(sigma, -1)

        shape = train.element_spec[0].shape[1:]

        out_shape = (configs["pred_len"],)
        if base_model == "transformer":
            base = transformer(shape, configs["chans"],configs["head_size"], configs["n_heads"], configs["features"])
        elif base_model == "LSTM":
            #out_shape = (configs["chans"], configs["pred_len"])
            base = lstm_encdec(configs["width"],configs["chans"],configs["layers"], 0.5, shape)
        elif base_model == "linear":
            base = linear(configs["input_channels"], configs["seq_len"],n_variates=configs["chans"])
        elif base_model == "independent_dense":
            base = independent_dense(configs["chans"], configs["seq_len"])
        else:
            base = dependent_dense(configs["chans"], configs["seq_len"])
        
        mse = tf.keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(configs["lr"])
        
        loss_model = None  
        if loss == "HL":
            loss_model = HLGaussian(base, borders, sigma, out_shape=out_shape)    
        else:
            loss_model = Regression(base, out_shape=out_shape)    
        wandb.init(config=configs, project="hl_loss_results")
        training(loss_model,train,test,configs["epochs"],optimizer,configs["pred_len"],mse) 
        wandb.finish()
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
