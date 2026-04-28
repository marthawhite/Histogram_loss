"""Module containing the run code for time series experiments.

Usage: 
    python main.py data_path

Params:
    data_path - path to the input data file
"""

import tensorflow as tf
from tensorflow import keras
from experiment.models import HLGaussian, Regression
from time_series.base_models import transformer, transformer_large, linear, lstm_encdec
import json
from experiment.bins import get_bins
from time_series.datasets import get_time_series_dataset
import wandb
import sys
from wandb.integration.keras import WandbMetricsLogger


class WandbValLogger(keras.callbacks.Callback):
    """Log validation mse/mae to wandb after Keras has already run validation.
    Reads from the model's compiled metric objects so no extra evaluate call
    is needed."""

    def on_epoch_end(self, epoch, logs=None):
        val_metrics = {}
        for m in self.model.metrics:
            result = m.result()
            if isinstance(result, dict):
                for key, value in result.items():
                    val_metrics[f"val_{key}"] = float(value)
            else:
                val_metrics[f"val_{m.name}"] = float(result)
        if val_metrics:
            wandb.log(val_metrics, commit=False)


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
    "pred_len" : 96,
    "seq_len" : 336,
    "epochs" : 100,
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
    "univariate":True, ## code is only doing univariate for now
    # transformer_large configs
    "d_model" : 256,
    "n_heads_large" : 8,
    "ff_dim" : 512,
    "n_layers" : 4,
    "dropout" : 0.2,
    "weight_decay" : 1e-4,
    "patience" : 20,
    }
    for dataset in configs["datasets"]:
        configs["dataset"] = dataset
        keras.utils.set_random_seed(1)
        data_path = f"{dataset}.csv"
        train, val, test, dmin, dmax = get_time_series_dataset(data_path, configs["drop"], configs["seq_len"], configs["batch_size"], configs["chans"], configs["input_target_offset"],configs["univariate"], pred_len=configs["pred_len"])
        pred_len = configs["pred_len"]
        train = train.map(lambda x, y: (x, tf.reshape(y, [-1, pred_len])))
        val = val.map(lambda x, y: (x, tf.reshape(y, [-1, pred_len])))
        test = test.map(lambda x, y: (x, tf.reshape(y, [-1, pred_len])))

        borders, sigma = get_bins(configs["n_bins"], configs["pad_ratio"], configs["sig_ratio"], dmin, dmax)
        # For multi-step prediction, add a trailing dimension so borders
        # broadcast correctly over (batch, pred_len) targets.
        # borders: (n_bins+1,) -> (n_bins+1, 1)
        if pred_len > 1:
            borders = tf.expand_dims(borders, -1)
        

        shape = (configs["seq_len"], train.element_spec[0].shape[-1])
        out_shape = () if configs["chans"] == 1 else (configs["chans"],)

        if base_model == "transformer_large":
            base = transformer_large(shape, configs["pred_len"],
                                     d_model=configs["d_model"],
                                     n_heads=configs["n_heads_large"],
                                     ff_dim=configs["ff_dim"],
                                     n_layers=configs["n_layers"],
                                     dropout=configs["dropout"])
        elif base_model == "transformer":
            base = transformer(shape, configs["pred_len"], configs["head_size"], configs["n_heads"], configs["features"])
        elif base_model == "LSTM":
            base = lstm_encdec(configs["width"], configs["pred_len"], configs["layers"], 0.5, shape)
        elif base_model == "linear":
            base = linear(configs["input_channels"], configs["seq_len"], pred_len=configs["pred_len"])
        elif base_model == "independent_dense":
            base = independent_dense(configs["chans"], configs["seq_len"])
        else:
            base = dependent_dense(configs["chans"], configs["seq_len"])
        print(base_model,configs["dataset"])
        configs["tag"] = f"{base_model}_{loss}_{dataset}"
        metrics = ["mse", "mae"]
        train_steps = len(train) // configs["batch_size"]

        # cosine decay 
        total_steps = configs["epochs"] * train_steps
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=configs["lr"],
            decay_steps=total_steps,
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=configs["weight_decay"],
        )

        # callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=configs["patience"],
            restore_best_weights=True,
            verbose=1,
        )

        if loss == "L2":
            configs["loss"] = 'regression'
            wandb.init(config=configs, project="hl_loss_results")
            callbacks = [WandbMetricsLogger(), early_stop]
            regression = Regression(base, out_shape=out_shape)
            regression.compile(optimizer=optimizer, loss="mse", metrics=metrics)
            for x, _ in train.take(1):
                regression(x)
            wandb.run.summary["num_params"] = regression.count_params()
            regression_history = regression.fit(x=train, epochs=configs["epochs"], steps_per_epoch=train_steps, validation_data=val, verbose=2, callbacks=callbacks)
            reg_results = regression.evaluate(test, return_dict=True, verbose=2)
            with open(f"reg_{dataset}.json", "w") as file:
                json.dump(regression_history.history, file)

            wandb.run.summary["test_loss"] = reg_results['loss']
            wandb.run.summary["test_mse"] = reg_results['mse']
            wandb.run.summary["test_mae"] = reg_results['mae']
            wandb.finish()
        elif loss == "HL":
            configs["loss"] = 'HL'
            wandb.init(config=configs, project="hl_loss_results")
            callbacks = [WandbMetricsLogger(), WandbValLogger(), early_stop]
            ## Run HL-Gaussian
            hl_gaussian = HLGaussian(base, borders, sigma, out_shape=out_shape)
            hl_gaussian.compile(optimizer=optimizer, metrics=metrics)
            for x, _ in train.take(1):
                hl_gaussian(x)
            wandb.run.summary["num_params"] = hl_gaussian.count_params()
            print("num_params", hl_gaussian.count_params())
            hl_gaussian_history = hl_gaussian.fit(x=train, epochs=configs["epochs"], steps_per_epoch=train_steps, validation_data=val, verbose=2, callbacks=callbacks)
            hl_results = hl_gaussian.evaluate(test, return_dict=True, verbose=2)
            with open(f"hlg_{dataset}.json", "w") as file:
                json.dump(hl_gaussian_history.history, file)

            wandb.run.summary["test_loss"] = hl_results['loss']
            wandb.run.summary["test_mse"] = hl_results['compile_metrics']['mse']
            wandb.run.summary["test_mae"] = hl_results['compile_metrics']['mae']
            wandb.finish()
        else:
            raise ValueError("Loss not recognized")
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])