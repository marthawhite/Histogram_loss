import sys
import os
from experiment.datasets import CSVDataset
from tensorflow import keras
from experiment.new_models import *
from experiment.hypermodels import *
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
    if dataset.name == "bike":
        return mlp_base(16, int_dim=4)


def get_datasets(base_dir):
    data_dir = os.path.join(base_dir, "data")
    ctscan = CSVDataset(os.path.join(data_dir, "slice_localization_data.csv"), "reference", batch_size=256)
    ctscan.set_bounds(0, 100)
    ctscan.name = "ctscan"

    bikeshare = CSVDataset(os.path.join(data_dir, "hour.csv"), "cnt", drop="dteday", batch_size=256)
    bikeshare.set_bounds(0, 1000)
    bikeshare.name = "bike"

    return [bikeshare]

def get_models(dataset):
    y_min, y_max = dataset.get_bounds()
    base = lambda : base_models(dataset)
    metrics = ["mse", "mae"]
    hp = kt.HyperParameters()
    hp.Fixed("dropout", 0)

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



def run(seeds, datasets, epochs):
    test_ratio = 0.2
    results = {}
    for dataset in datasets:
        models = get_models(dataset)
        for model in models:
            results[model.name] = []
        for seed in seeds:
            keras.utils.set_random_seed(seed)
            train, test = dataset.get_split(test_ratio, True)
            for model in models:
                print(dataset.name, seed, model.name)
                hist = model.fit(train, epochs=epochs)
                outputs = model.evaluate(test, return_dict=True)
                results[model.name].append({
                    "train_loss": hist.history["loss"][-1],
                    "train_mse": hist.history["mse"][-1],
                    "train_mae": hist.history["mae"][-1],
                    "test_loss": outputs["loss"],
                    "test_mse": outputs["mse"],
                    "test_mae": outputs["mae"]
                })
    return results


def save(outfile, results):
    with open(outfile, "w") as out_file:
        json.dump(results, out_file, indent=4)


def main(base_dir):
    epochs = 10
    seeds = [1]
    outfile = "test_exp.json"
    datasets = get_datasets(base_dir)
    results = run(seeds, datasets, epochs)
    save(outfile, results)


if __name__ == "__main__":
    data_file = sys.argv[1]
    main(data_file)