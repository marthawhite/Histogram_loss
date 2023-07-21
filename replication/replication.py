"""Module for running replication experiments.

Based on previous papers and code by Ehsan Imani.

Usage: python replication.py data_dir

Params:
    data_dir - the directory containing the CSV files for the datasets
"""

import sys
import os
from replication.csvdataset import CSVDataset
from tensorflow import keras
from experiment.models import *
from experiment.hypermodels import *
from experiment.preprocessing import *
import keras_tuner as kt
import json


def mlp_base(input_width, hidden=4, dropout=0.05, int_dim=0.5):
    """Return an MLP base model to learn features from the data.
    
    Params:
        input_width - the number of input features
        hidden - the number of hidden layers
        dropout - the dropout rate to use on the input
        int_dim - the ratio of the size of the hidden layers relative to the input size
    
    Returns: a Keras model 
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(dropout))
    width = int(int_dim * input_width)
    for i in range(hidden):
        model.add(keras.layers.Dense(width, activation='relu', kernel_initializer='lecun_uniform'))
    return model


def base_models(dataset):
    """Return the base model for a given dataset.
    
    Params:
        dataset - the name of the dataset

    Returns: the MLP base model
    """
    if dataset.name == "ctscan":
        return mlp_base(385)
    elif dataset.name == "bike":
        return mlp_base(16, int_dim=4, dropout=0)
    elif dataset.name == "pole":
        return mlp_base(49, dropout=0)
    elif dataset.name == "songyear":
        return mlp_base(90)


def get_datasets(base_dir):
    """Load the datasets to use in the experiment.
    
    Params:
        base_dir - the directory in which the CSV files are located

    Returns: a list of dataset objects
    """
    data_dir = os.path.join(base_dir, "data")
    ctscan = CSVDataset(os.path.join(data_dir, "slice_localization_data.csv"), "reference", batch_size=256)
    ctscan.bounds = (0., 100.)
    ctscan.name = "ctscan"
    ctscan.epochs = 1000

    bikeshare = CSVDataset(os.path.join(data_dir, "hour.csv"), "cnt", drop_cols="dteday", batch_size=256)
    bikeshare.bounds = (0., 1000.)
    bikeshare.name = "bike"
    bikeshare.epochs = 500

    songyear = CSVDataset(os.path.join(data_dir, "YearPredictionMSD.txt"), 0, header=None, batch_size=256)
    songyear.bounds = (1922., 2011.)
    songyear.name = "songyear"
    songyear.epochs = 150

    pole = CSVDataset(os.path.join(data_dir, "pole.csv"), "target", batch_size=256)
    pole.bounds = (0., 100.)
    pole.name = "pole"
    pole.epochs = 500

    return [ctscan, bikeshare, songyear, pole]

def get_models(dataset, scale=True):
    """Get the models for a given dataset.
    
    Params:
        dataset - a Dataset object with the data and info
        scale - True if the y data will be scaled to [0, 1]
    
    Returns: a list of models to run on the dataset
    """
    if scale:
        y_min, y_max = 0., 1.
    else:
        y_min, y_max = dataset.bounds
    base = lambda : base_models(dataset)
    metrics = ["mse", "mae"]
    hp = kt.HyperParameters()
    hp.Fixed("dropout", 0)
    hp.Fixed("padding", 0.125)
    hp.Fixed("sig_ratio", 1.)
    hp.Fixed("n_bins", 100)
    hp.Fixed("learning_rate", 1e-3)

    hyperhlg = HyperHLGaussian(base, y_min, y_max, metrics=metrics)
    hlg = hyperhlg.build(hp)

    hyperl2 = HyperRegression(base, loss="mse", metrics=metrics, name="L2")
    l2 = hyperl2.build(hp)

    hyperl1 = HyperRegression(base, loss="mae", metrics=metrics, name="L1")
    l1 = hyperl1.build(hp)

    hyperhl1 = HyperHLOneBin(base, y_min, y_max, metrics=metrics)
    hl1 = hyperhl1.build(hp)

    lin_base = lambda : keras.layers.Identity()
    hyperlin = HyperRegression(lin_base, loss="mse", metrics=metrics, name="LinReg")
    lin = hyperlin.build(hp)

    return [l1, l2, hlg, hl1, lin]


def run_model(model, epochs, train, test):
    """Run an experiment for one model on a dataset.
    
    Params:
        model - the Keras model to test
        epochs - the number of epochs to train for
        train - the tf Dataset with the training split
        test - the tf Dataset with the testing split

    Returns: results - a dict of the training and testing metrics
    """
    hist = model.fit(train, epochs=epochs, verbose=2)
    outputs = model.evaluate(test, return_dict=True, verbose=2)
    results = {
        "train_loss": hist.history["loss"][-1],
        "train_mse": hist.history["mse"][-1],
        "train_mae": hist.history["mae"][-1],
        "test_loss": outputs["loss"],
        "test_mse": outputs["mse"],
        "test_mae": outputs["mae"]
    }
    return results


def preprocess(train, test, bounds, scale, norm):
    """Preprocess the data by scaling and normalizing.
    
    Params:
        train - the tf Dataset with the train split
        test - the tf Dataset with the test split
        bounds - a tuple with the minimum and maximum y values
        scale - True if the y values will be scaled to [0, 1]; False otherwise
        norm - True if the x values will be normalized based on the training data; False otherwise
    
    Returns: (train, test) - the transformed train and test splits
    """
    if scale:
        sc = Scaler(*bounds)
        train = sc.transform(train)
        test = sc.transform(test)

    if norm:
        norm = Normalizer()
        norm.fit(train)
        train = norm.transform(train)
        test = norm.transform(test)
    return train, test

def run_seed(dataset, seed, test_ratio, scale=True, norm=True):
    """Run the experiment on a dataset for all models with a given seed.
    
    Params:
        dataset - the Dataset object to run the experiment on
        seed - the seed to use for the experiment
        test_ratio - the proportion of samples held out for testing
        scale - True if the y values will be scaled to [0, 1]; False otherwise
        norm - True if the x values will be normalized based on the training data; False otherwise

    Returns: results - a dict with the results for each model
    """
    results = {}
    keras.utils.set_random_seed(seed)
    models = get_models(dataset)
    train, test = dataset.get_split(test_ratio, shuffle=True)

    train, test = preprocess(train, test, dataset.bounds, scale, norm)

    for model in models:
        results[model.name] = run_model(model, dataset.epochs, train, test)
    return results


def run_dataset(dataset, seeds, test_ratio):
    """Run an experiment on a dataset with multiple seeds.
    
    Params:
        dataset - the Dataset object to run the experiment on
        seeds - the list of seeds to use
        test_ratio - the proportion of samples held out for testing

    Returns: results - a dict with the results for each seed
    """
    results = {}
    for seed in seeds:
        results[seed] = run_seed(dataset, seed, test_ratio)
        outfile = os.path.join("temp_results", f"{dataset.name}-{seed}.json")
        save(outfile, results)
    return results


def run(seeds, datasets, test_ratio):
    """Run the experiment on multiple datasets and seeds.
    
    Params:
        seeds - the list of seeds to use
        datasets - the list of Datasets to use
        test_ratio - the proportion of samples held out for testing 

    Returns: results - a dict with the results for each dataset
    """
    results = {}
    for dataset in datasets:
        results[dataset.name] = run_dataset(dataset, seeds, test_ratio) 
    return results


def save(outfile, results):
    """Save the results of the experiment to a JSON file.
    
    Params:
        outfile - the file to save the results to
        results - the dict with the results to save
    """
    with open(outfile, "w") as out_file:
        json.dump(results, out_file, indent=4)


def main(base_dir):
    """Run the replication experiment."""
    test_ratio = 0.2
    seeds = [1, 2, 3, 4, 5]
    outfile = "replication.json"
    datasets = get_datasets(base_dir)
    results = run(seeds, datasets, test_ratio)
    save(outfile, results)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)