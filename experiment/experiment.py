from experiment.logging import LogGridSearch
from tensorflow import keras
import json

class HyperExperiment:

    def __init__(self, models, dataset, outfile, search_args, fit_args, test_ratio=None) -> None:
        self.models = models
        self.ds = dataset
        self.outfile = outfile
        self.search_args = search_args
        self.fit_args = fit_args
        self.test_ratio = test_ratio

    def run(self):
        results = {}
        train, test = self.ds.get_split(self.test_ratio)
        for model in self.models:
            tuner = LogGridSearch(
                hypermodel=model,
                project_name=model.name,
                json_file= "temp_" + self.outfile,
                **self.search_args
            )
            tuner.search(x=train, validation_data=test, **self.fit_args)
            results[model.name] = tuner.get_results()

        with open(self.outfile, "w") as out_file:
            json.dump(results, out_file, indent=4)
