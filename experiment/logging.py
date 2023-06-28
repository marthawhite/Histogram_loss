"""
LogGridSearch class.
"""


import keras_tuner as kt
import json


class LogGridSearch(kt.GridSearch):
    """Tuner class that stores trial results in a dict.
    
    Params:
        kwargs - arguments for GridSearch class
            ***Should contain metrics!
    """

    def __init__(self, json_file="temp_results.json", **kwargs):
        super().__init__(**kwargs)
        self.logs = {}
        self.metric_list = ["loss", "val_loss"]
        for key in self.metrics:
            self.metric_list.append(key)
            self.metric_list.append("val_" + key)
        self.out_file = json_file

    def on_trial_begin(self, trial):
        """Initialize the dict entry when the trial begins.
        
        Params:
            trial - the Trial instance; contains hyperparameters
        """
        super().on_trial_begin(trial)
        self.logs[trial.trial_id] = {}
        self.logs[trial.trial_id]["hypers"] = trial.hyperparameters.values
        self.logs[trial.trial_id]["results"] = []
        self.ex_num = -1

    def init_run(self):
        empty = {}
        for key in self.metric_list:
            empty[key] = []
        return empty

    def on_epoch_end(self, trial, model, epoch, logs=None):
        """Update the logs when a batch completes.
        
        Params:
            trial - the Trial instance
            logs - the results dict from model.fit()
        """
        super().on_batch_end(trial, model, epoch, logs)
        if epoch == 0:
            self.ex_num += 1
            self.logs[trial.trial_id]["results"].append(self.init_run())
        for key in self.metric_list:
            self.logs[trial.trial_id]["results"][self.ex_num][key].append(logs.get(key, None))

    def on_trial_end(self, trial):
        """Save intermediate results at the end of each trial."""
        super().on_trial_end(trial)
        self.save_results()

    def save_results(self):
        """Save results to a json file."""
        with open(self.out_file, "w") as out_file:
            json.dump(self.logs, out_file, indent=4)

    def get_results(self):
        """Return the results as a dictionary."""
        return self.logs