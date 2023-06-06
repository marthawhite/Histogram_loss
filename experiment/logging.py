import keras_tuner as kt


class LogGridSearch(kt.GridSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logs = {}
        self.metric_list = []
        for key in self.metrics:
            self.metric_list.append(key)
            self.metric_list.append("val_" + key)

    def on_trial_begin(self, trial):
        super().on_trial_begin(trial)
        self.logs[trial.trial_id] = {}
        self.logs[trial.trial_id]["hypers"] = trial.hyperparameters.values
        self.logs[trial.trial_id]["results"] = {}
        for key in self.metric_list:
            self.logs[trial.trial_id]["results"][key] = []

    def on_epoch_end(self, trial, model, epoch, logs=None):
        super().on_batch_end(trial, model, epoch, logs)
        for key in self.metric_list:
            self.logs[trial.trial_id]["results"][key].append(logs.get(key, None))