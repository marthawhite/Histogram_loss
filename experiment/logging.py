"""
LogGridSearch class.
"""


import keras_tuner as kt


class LogGridSearch(kt.GridSearch):
    """Tuner class that stores trial results in a dict.
    
    Params:
        kwargs - arguments for GridSearch class
            ***Should contain metrics!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logs = {}
        self.metric_list = ["loss", "val_loss"]
        for key in self.metrics:
            self.metric_list.append(key)
            self.metric_list.append("val_" + key)

    def on_trial_begin(self, trial):
        """Initialize the dict entry when the trial begins.
        
        Params:
            trial - the Trial instance; contains hyperparameters
        """
        super().on_trial_begin(trial)
        self.logs[trial.trial_id] = {}
        self.logs[trial.trial_id]["hypers"] = trial.hyperparameters.values
        self.logs[trial.trial_id]["results"] = {}
        for key in self.metric_list:
            self.logs[trial.trial_id]["results"][key] = []

    def on_epoch_end(self, trial, model, epoch, logs=None):
        """Update the logs when a batch completes.
        
        Params:
            trial - the Trial instance
            logs - the results dict from model.fit()
        """
        super().on_batch_end(trial, model, epoch, logs)
        for key in self.metric_list:
            self.logs[trial.trial_id]["results"][key].append(logs.get(key, None))