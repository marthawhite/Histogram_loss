"""
Utility module to convert experiment results.

Run configuration:
    python3 jsontocsv.py in_file1.json ... in_fileN.json out_file.csv
"""

import pandas as pd
import json
import sys


class JSONCSVConverter:
    """Convert experiment results from JSON to CSV.
    
    The JSON files should have the following structure:
    {
        "model1": {
            "trial1": {
                "hypers": { ... },
                "results": { ... }
            },
            ...
        },
        ...
    }
    """

    def convert(self, in_files, out_file, cols=None):
        """Create a CSV file from JSON results.
        
        Params:
            in_files - a list of paths to the JSON files
            out_file - the path to the CSV file to create
            cols - the list of column names to specify an order
        """
        dfs = []
        for in_file in in_files:
            data = self.read(in_file)
            df = self.transform(data)
            dfs.append(df)
        df = pd.concat(dfs)
        if cols is not None:
            df = self.reorder(df, cols)
        self.write(out_file, df)

    def read(self, in_file):
        """Read that data from a JSON file.
        
        Params:
            in_file - the path to the JSON 

        Returns: the dict containing the data
        """
        with open(in_file, "r") as json_file:
            data = json.load(json_file)
        return data
    
    def get_trial(self, name, trial):
        """Create a dataframe from a trial instance.
        
        Params:
            name - the name of the trial
            trial - a dict containing the trial information
        
        Returns: a Pandas Dataframe containing rows for each epoch in the trial
        """
        dfs = []
        for i, res in enumerate(trial["results"]):
            df = pd.DataFrame.from_dict(res)
            df['iteration'] = i + 1
            dfs.append(df)
        df = pd.concat(dfs)
        for hyper, hyper_val in trial["hypers"].items():
            df[hyper] = hyper_val
        df["trial"] = name
        return df

    def get_model(self, name, model):
        """Create a dataframe for a model experiment.
        
        Params:
            name - the name of the model used
            model - a dict containing the results from the model

        Returns: a Pandas Dataframe containing rows for each epoch of each trial
        """
        dfs = []
        for trial, trial_d in model.items():
            df = self.get_trial(trial, trial_d)
            dfs.append(df)
        new_df = pd.concat(dfs)
        new_df["model"] = name
        return new_df

    def transform(self, data):
        """Transform the data into a Pandas Dataframe.
        
        Params:
            data - the dict containing experiment results
        
        Returns: a Pandas Dataframe with one row per epoch tested
        """
        models = []
        for model, model_d in data.items():
            new_df = self.get_model(model, model_d)
            models.append(new_df)
        final = pd.concat(models)
        final["epoch"] = final.index + 1
        return final
    
    def reorder(self, df, cols):
        """Reorder the columns of df.
        
        Params:
            df - the dataframe to reorder
            cols - a list containing the first columns of the new dataframe
                Other columns will appear in an arbitrary order afterwards
        
        Returns: a dataframe with reordered columns
        """
        last_cols = [col for col in df.columns if col not in cols]
        return df[cols + last_cols]
    
    def write(self, out_file, df):
        """Write the dataframe to a csv file.
        
        Params:
            out_file - a path to a new CSV file
            df - the dataframe to write
        """
        df.to_csv(out_file, index=False)


def main():
    in_files = sys.argv[1:-1]
    out_file = sys.argv[-1]
    conv = JSONCSVConverter()
    cols = ["model", "trial", "iteration", "epoch"]
    conv.convert(in_files, out_file, cols)


if __name__ == "__main__":
    main()
