from experiment.logging import LogGridSearch
from tensorflow import keras
import json


class Experiment:

    def __init__(self, name="Experiment") -> None:
        self.results = {}
        self.name = name

    def run(self):
        pass

    def save(self, outfile):
        with open(outfile, "w") as out_file:
            json.dump(self.results, out_file, indent=4)


class HyperExperiment(Experiment):

    def __init__(self, model, dataset, search_args, fit_args, test_ratio=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hmodel = model
        self.ds = dataset
        self.search_args = search_args
        self.fit_args = fit_args
        self.test_ratio = test_ratio

    def run(self):
        train, test = self.ds.get_split(self.test_ratio)
        tuner = LogGridSearch(
            hypermodel=self.hmodel,
            project_name=self.hmodel.name,
            **self.search_args
        )
        tuner.search(x=train, validation_data=test, **self.fit_args)
        self.results = tuner.get_results()
        return self.results


class ConcreteExperiment(Experiment):

    def __init__(self, model, dataset, test_ratio, fit_args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.ds = dataset
        self.test_ratio = test_ratio
        self.fit_args = fit_args

    def run(self):
        train, test = self.ds.get_split(self.test_ratio)
        hist = self.model.fit(train, validation_data=test, **self.fit_args)
        self.results = hist.history
        return self.results


class MultiExperiment(Experiment):

    def __init__(self, experiments, **kwargs) -> None:
        super().__init__(**kwargs)
        self.experiments = experiments

    def run(self):
        for exp in self.experiments:
            self.results[exp.name] = exp.run()
        return self.results
