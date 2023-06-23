from experiment.logging import LogGridSearch
from tensorflow import keras
import json

class BaseExperiment:

    def __init__(self) -> None:
        self.results = {}

    def run(self):
        return self.results

    def save(self, filename):
        with open(filename, "w") as out_file:
            json.dump(self.results, out_file, indent=4)


class Experiment(BaseExperiment):

    def __init__(self, model, dataset) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
    
    def run(self, split_args):
        train, val, test = self.dataset.three_split(val_ratio, test_ratio)
        self.model.fit(train, validation_data=val)


class HyperExperiment(BaseExperiment):

    def __init__(self, model, dataset, search_args, fit_args, test_ratio=None) -> None:
        self.model = model
        self.ds = dataset
        self.search_args = search_args
        self.fit_args = fit_args
        self.test_ratio = test_ratio

    def run(self):
        results = {}
        train, test = self.ds.get_split(self.test_ratio)
        tuner = LogGridSearch(
            hypermodel=self.model,
            project_name=self.model.name,
            json_file= "temp_" + self.outfile,
            **self.search_args
        )
        tuner.search(x=train, validation_data=test, **self.fit_args)
        results[self.model.name] = tuner.get_results()
        
