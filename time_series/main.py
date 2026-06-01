"""Module containing the run code for time series experiments.

Usage: 
    python main.py data_path

Params:
    data_path - path to the input data file
"""

import tensorflow as tf
from tensorflow import keras
from experiment.models import HLGaussian, Regression
from time_series.base_models import transformer, linear, lstm_encdec, gru_linear, independent_dense, dependent_dense
from time_series.autoformer import autoformer
import json
import numpy as np
from experiment.bins import get_bins
from time_series.datasets import get_time_series_dataset
import wandb
import sys
from wandb.integration.keras import WandbMetricsLogger
from dataclasses import dataclass, field, asdict
from typing import Literal
import tyro


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


@dataclass
class Config:
    """Time-series experiment configuration """ 
    base_model: Literal["transformer", "autoformer", "LSTM", "GRU",
                        "linear"] = "LSTM"
    loss: Literal["HL", "L2"] = "HL"
    seed: int = 1
    # data
    datasets: list[str] = field(
        default_factory=lambda: ["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    seq_len: int = 96
    pred_len: int = 96
    input_channels: int = 7
    chans: int = 1                  # number of target prediction variables
    input_target_offset: int = 0
    drop: str = "date"
    univariate: bool = True         
    test_ratio: float = 0.25        # split is month-based
    # optimizer
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4      # AdamW
    patience: int = 3
    # HL-Gaussian
    n_bins: int = 100
    sig_ratio: float = 2.0
    pad_ratio: float = 3.0
    # vanilla transformer
    tf_d_model: int = 128
    tf_n_heads: int = 8
    tf_ff_dim: int = 512
    tf_blocks: int = 3
    # LSTM / GRU
    layers: int = 2
    width: int = 256
    rnn_dropout: float = 0.1
    # autoformer
    d_model: int = 256
    n_heads_large: int = 8
    ff_dim: int = 512
    dropout: float = 0.05
    moving_avg: int = 25
    factor: int = 1
    activation: str = "relu"


def main(cfg: Config):
    """Run the time series experiment.

    Params:
        base_model(str):  Name of the base model
        loss(str): Name of loss, either HL or L2
        seed(int): Random seed controlling weight init, dropout, and the
            training-data shuffle.
        
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
            tf_d_model: Model hidden dimension
            tf_n_heads: Number of attention heads (key_dim = tf_d_model // tf_n_heads)
            tf_ff_dim: Feed-forward width (set to 4 * tf_d_model)
            tf_blocks: Number of encoder blocks
        LSTM / GRU:
            layers: Number of stacked recurrent layers
            width: Width of the recurrent layers
            rnn_dropout: Recurrent dropout rate
        
        
    """
    configs = asdict(cfg)
    base_model = cfg.base_model
    loss = cfg.loss
    seed = cfg.seed
    for dataset in configs["datasets"]:
        configs["dataset"] = dataset
        keras.utils.set_random_seed(seed)
        data_path = f"{dataset}.csv"
        train, val, test, dmin, dmax = get_time_series_dataset(
            data_path,
            drop=configs["drop"],
            seq_len=configs["seq_len"],
            batch_size=configs["batch_size"],
            input_target_offset=configs["input_target_offset"],
            pred_len=configs["pred_len"],
            shuffle_seed=seed,
        )
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

        if base_model == "autoformer":
            base = autoformer(shape, configs["pred_len"],
                              d_model=configs["d_model"],
                              n_heads=configs["n_heads_large"],
                              d_ff=configs["ff_dim"],
                              e_layers=2, d_layers=1,
                              factor=configs["factor"],
                              moving_avg=configs["moving_avg"],
                              dropout=configs["dropout"],
                              activation=configs["activation"],
                              c_out=configs["d_model"])
        elif base_model == "transformer":
            base = transformer(shape, configs["pred_len"],
                               d_model=configs["tf_d_model"],
                               n_heads=configs["tf_n_heads"],
                               ff_dim=configs["tf_ff_dim"],
                               n_blocks=configs["tf_blocks"],
                               dropout=configs["dropout"])
        elif base_model == "LSTM":
            base = lstm_encdec(configs["width"], configs["pred_len"], configs["layers"], configs["rnn_dropout"], shape)
        elif base_model == "GRU":
            base = gru_linear(configs["width"], configs["pred_len"], configs["layers"], configs["rnn_dropout"], shape)
        elif base_model == "linear":
            base = linear(configs["input_channels"], configs["seq_len"], pred_len=configs["pred_len"])
        elif base_model == "independent_dense":
            base = independent_dense(configs["chans"], configs["seq_len"])
        else:
            base = dependent_dense(configs["chans"], configs["seq_len"])
        print(base_model,configs["dataset"])
        configs["tag"] = f"{base_model}_{loss}_{dataset}_s{seed}"
        metrics = ["mse", "mae"]
        train_steps = len(train) # len of train returns the number of batches

        optimizer = keras.optimizers.AdamW(
            learning_rate=configs["lr"],
            weight_decay=configs["weight_decay"],
        )
        # callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_mse",
            mode="min",
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
            with open(f"reg_{base_model}_{dataset}_s{seed}.json", "w") as file:
                json.dump(regression_history.history, file)

            wandb.run.summary["test_loss"] = reg_results['loss']
            wandb.run.summary["test_mse"] = reg_results['mse']
            wandb.run.summary["test_mae"] = reg_results['mae']

            # Save val predictions, targets, and input context for plotting
            val_preds, val_targets, val_inputs = [], [], []
            for x_batch, y_batch in val:
                val_preds.append(regression(x_batch, training=False).numpy())
                val_targets.append(y_batch.numpy())
                val_inputs.append(x_batch[:, :, -1].numpy())  # last channel (OT)
            np.save(f"{base_model}_{loss}_{dataset}_{seed}_val_preds.npy", np.concatenate(val_preds))
            np.save(f"{base_model}_{loss}_{dataset}_{seed}_val_targets.npy", np.concatenate(val_targets))
            np.save(f"{base_model}_{loss}_{dataset}_{seed}_val_inputs.npy", np.concatenate(val_inputs))

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
            with open(f"hlg_{base_model}_{dataset}_s{seed}.json", "w") as file:
                json.dump(hl_gaussian_history.history, file)

            wandb.run.summary["test_loss"] = hl_results['loss']
            wandb.run.summary["test_mse"] = hl_results['mse']
            wandb.run.summary["test_mae"] = hl_results['mae']

            # Save val predictions, targets, and input context for plotting
            val_preds, val_targets, val_inputs = [], [], []
            for x_batch, y_batch in val:
                val_preds.append(hl_gaussian(x_batch, training=False).numpy())
                val_targets.append(y_batch.numpy())
                val_inputs.append(x_batch[:, :, -1].numpy())  # last channel (OT)
            np.save(f"{base_model}_{loss}_{dataset}_{seed}_val_preds.npy", np.concatenate(val_preds))
            np.save(f"{base_model}_{loss}_{dataset}_{seed}_val_targets.npy", np.concatenate(val_targets))
            np.save(f"{base_model}_{loss}_{dataset}_{seed}_val_inputs.npy", np.concatenate(val_inputs))

            wandb.finish()
        else:
            raise ValueError("Loss not recognized")
if __name__ == "__main__":
    main(tyro.cli(Config))