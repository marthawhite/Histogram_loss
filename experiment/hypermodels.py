"""Module containing hypermodels to use for hyperparameter tuning."""


import keras_tuner as kt
from experiment.models import Regression, HLGaussian, HLOneBin, HLUniform, HLProjected
from tensorflow import keras
import tensorflow as tf


class HyperBase(kt.HyperModel):
    """Hypermodel that builds and returns compiled Keras models. 
    
    Params:
        name - name of the hypermodel
        loss - the loss to compile the model with
        metrics - the metrics to compile the model with
    """

    def __init__(self, name="HyperBase", loss=None, metrics=None):
        super().__init__(name, True)
        self.loss = loss
        self.metrics = metrics

    def build(self, hp):
        """Build and return a Keras model.
        
        Params:
            hp - the KerasTuner HyperParameter instance to use to build the model
        """
        model = self.get_model(hp)
        model.compile(optimizer=self.get_opt(hp), loss=self.loss, metrics=self.metrics)
        return model
    
    def get_opt(self, hp):
        """Return the optimizer to use to compile the model.
        
        Params:
            hp - the KerasTuner HyperParameter instance
        """
        lr = hp.Float("learning_rate", default=1e-3, min_value=1e-4, max_value=1e-2, step=10, sampling="log")
        b1 = hp.Fixed("beta_1", 0.9)
        b2 = hp.Fixed("beta_2", 0.999)
        eps = hp.Fixed("epsilon", 1e-7)
        opt = keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)
        return opt

    def get_model(self, hp):
        """Return the Keras model"""
        pass
    

class HyperRegression(HyperBase):
    """Hypermodel that adds a regression layer to a base model.

    Params:
        base - the base keras.Model
        loss - the loss function to compile the model with
        metrics - the metrics to compile the model with
    """

    def __init__(self, base, loss=None, metrics=None, name="HyperReg"):
        super().__init__(name, loss, metrics)
        self.base = base

    def get_model(self, hp):
        """Return a regression model."""
        dropout = hp.Choice("dropout", [0., 0.2, 0.5, 0.8], default=0.5)
        return Regression(self.base(), dropout, self.name)
    

class HyperHL(HyperBase):
    """Hypermodel that adds a histogram loss layer to a base model.
    
    Params:
        min_y - the minimum target value
        max_y - the maximum target value
        name - the name of the hypermodel
        metrics - the metrics to compile the model with
    """

    def __init__(self, min_y, max_y, name="HyperHL", metrics=None):
        super().__init__(name, None, metrics)
        self.y_min = min_y
        self.y_max = max_y

    def get_model(self, hp):
        """Return the keras model."""
        pass

    def get_bins(self, hp):
        """Generate the bins given the minimum, maximum, and hyperparameters.
        
        Params:
            hp - the KerasTuner HyperParameter instance to use to build the model
        """
        padding = hp.Float("padding", default=0.1, min_value=0.025, max_value=0.1, step=2, sampling="log")
        n_bins = int(hp.Int("n_bins", default=100, min_value=25, max_value=400, step=2, sampling="log"))
        
        # Add padding proportional to the data range
        y_range = self.y_max - self.y_min
        new_min = self.y_min - padding * y_range
        new_max = self.y_max + padding * y_range
        bins = tf.linspace(new_min, new_max, n_bins)
        return bins


class HyperHLGaussian(HyperHL):
    """Histogram loss hypermodel that uses a truncated Gaussian distribution
    for its targets.
    
    Params:
        base - the base model
        min_y - the minimum target value
        max_y - the maximum target value
        metrics - the metrics to compile the model with
    """

    def __init__(self, base, min_y, max_y, metrics=None):
        super().__init__(min_y, max_y, "HyperHL-Gaussian", metrics)
        self.base = base

    def get_model(self, hp):
        """Return the HLGaussian model according to the hyperparameters.
        
        Params:
            hp - the KerasTuner HyperParameter instance
        """
        sig_ratio = hp.Float("sig_ratio", default=1., min_value=0.5, max_value=2., step=2, sampling="log")
        dropout = hp.Choice("dropout", [0., 0.2, 0.5, 0.8], default=0.5)

        # Calculate sigma as a multiple of the bin width
        bins = self.get_bins(hp)
        bin_width = bins[1] - bins[0]
        sigma = sig_ratio * bin_width
        return HLGaussian(self.base(), bins, sigma, dropout)
    

class HyperHLOneBin(HyperHL):
    """Histogram loss hypermodel using one-hot targets.
    
    Params:
        base - the base Keras model
        min_y - the minimum target value
        max_y - the maximum target value
        metrics - the metrics to compile the model with
    """

    def __init__(self, base, min_y, max_y, metrics=None):
        super().__init__(min_y, max_y, "HyperHL-OneBin", metrics)
        self.base = base

    def get_model(self, hp):
        """Return the HLOneBin model according to the hyperparameters.
        
        Params:
            hp - the KerasTuner HyperParameter instance
        """
        dropout = hp.Choice("dropout", [0., 0.2, 0.5, 0.8], default=0.5)
        bins = self.get_bins(hp)
        return HLOneBin(self.base(), bins, dropout)
    

class HyperHLUniform(HyperHL):
    """Histogram loss hypermodel using one-hot targets.
    
    Params:
        base - the base Keras model
        min_y - the minimum target value
        max_y - the maximum target value
        metrics - the metrics to compile the model with
    """

    def __init__(self, base, min_y, max_y, metrics=None):
        super().__init__(min_y, max_y, "HyperHL-Uniform", metrics)
        self.base = base

    def get_model(self, hp):
        """Return the HLUniform model according to the hyperparameters.
        
        Params:
            hp - the KerasTuner HyperParameter instance
        """
        dropout = hp.Choice("dropout", [0., 0.05, 0.2, 0.5], default=0.2)
        eps = hp.Fixed("eps", 1e-3)
        bins = self.get_bins(hp)
        return HLUniform(self.base(), bins, dropout, eps)


class HyperHLProjected(HyperHL):
    """Histogram loss hypermodel using projected targets.
    
    Params:
        base - the base Keras model
        min_y - the minimum target value
        max_y - the maximum target value
        metrics - the metrics to compile the model with
    """

    def __init__(self, base, min_y, max_y, metrics=None):
        super().__init__(min_y, max_y, "HyperHL-Projected", metrics)
        self.base = base

    def get_model(self, hp):
        """Return the HLUniform model according to the hyperparameters.
        
        Params:
            hp - the KerasTuner HyperParameter instance
        """
        dropout = hp.Choice("dropout", [0., 0.05, 0.2, 0.5], default=0.2)
        bins = self.get_bins(hp)
        return HLProjected(self.base(), bins, dropout)
