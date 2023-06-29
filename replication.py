import sys
import os
from experiment.datasets import CSVDataset
from tensorflow import keras
from experiment.models import *
from experiment.hypermodels import *
from experiment.preprocessing import *
import keras_tuner as kt
import json


def mlp_base(input_width, hidden=4, dropout=0.05, int_dim=0.5):
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(dropout))
    width = int(int_dim * input_width)
    for i in range(hidden):
        model.add(keras.layers.Dense(width, activation='relu', kernel_initializer='lecun_uniform'))
    return model


def base_models(dataset):
    if dataset.name == "ctscan":
        return mlp_base(385)
    elif dataset.name == "bike":
        return mlp_base(16, int_dim=4, dropout=0)
    elif dataset.name == "pole":
        return mlp_base(49, dropout=0)
    elif dataset.name == "songyear":
        return mlp_base(90)


def get_datasets(base_dir):
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
    results = {}
    keras.utils.set_random_seed(seed)
    models = get_models(dataset)
    train, test = dataset.get_split(test_ratio, shuffle=True)

    train, test = preprocess(train, test, dataset.bounds, scale, norm)

    for model in models:
        results[model.name] = run_model(model, dataset.epochs, train, test)
    return results


def run_dataset(dataset, seeds, test_ratio):
    results = {}
    for seed in seeds:
        results[seed] = run_seed(dataset, seed, test_ratio)
        outfile = os.path.join("temp_results", f"{dataset.name}-{seed}.json")
        save(outfile, results)
    return results


def run(seeds, datasets, test_ratio):
    results = {}
    for dataset in datasets:
        results[dataset.name] = run_dataset(dataset, seeds, test_ratio) 
    return results


def save(outfile, results):
    with open(outfile, "w") as out_file:
        json.dump(results, out_file, indent=4)


def main(base_dir):
    test_ratio = 0.2
    seeds = [1, 2, 3, 4, 5]
    outfile = "replication.json"
    datasets = get_datasets(base_dir)
    results = run(seeds, datasets, test_ratio)
    save(outfile, results)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)