import keras_tuner as kt
from experiment.new_models import Regression, HLGaussian, HLOneBin
from tensorflow import keras
import tensorflow as tf

class HyperBase(kt.HyperModel):

    def __init__(self, name="HyperBase", loss=None, metrics=None):
        super().__init__(name, True)
        self.loss = loss
        self.metrics = metrics

    def build(self, hp):
        model = self.get_model(hp)
        model.compile(optimizer=self.get_opt(hp), loss=self.loss, metrics=self.metrics)
        return model
    
    def get_opt(self, hp):
        lr = hp.Float("learning_rate", default=1e-3, min_value=1e-4, max_value=1e-2, step=10, sampling="log")
        b1 = hp.Fixed("beta_1", 0.9)
        b2 = hp.Fixed("beta_2", 0.999)
        eps = hp.Fixed("epsilon", 1e-7)
        opt = keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)
        return opt

    def get_model(self, hp):
        pass
    
class HyperRegression(HyperBase):

    def __init__(self, base, loss=None, metrics=None):
        super().__init__("HyperBase", loss, metrics)
        self.base = base

    def get_model(self, hp):
        return Regression(self.base())
    
class HyperHL(HyperBase):

    def __init__(self, min_y, max_y, name="HyperHL", metrics=None):
        super().__init__(name, None, metrics)
        self.y_min = min_y
        self.y_max = max_y

    def get_model(self, hp):
        pass

    def get_bins(self, hp):
        padding = hp.Float("padding", default=0.1, min_value=0.025, max_value=0.1, step=2, sampling="log")
        n_bins = int(hp.Int("n_bins", default=100, min_value=25, max_value=400, step=2, sampling="log"))
        
        y_range = self.y_max - self.y_min
        new_min = self.y_min - padding * y_range
        new_max = self.y_max + padding * y_range
        bins = tf.linspace(new_min, new_max, n_bins)
        return bins


class HyperHLGaussian(HyperHL):

    def __init__(self, base, min_y, max_y, metrics=None):
        super().__init__(min_y, max_y, "HyperHL-Gaussian", metrics)
        self.base = base

    def get_model(self, hp):
        sig_ratio = hp.Float("sig_ratio", default=1., min_value=0.5, max_value=2., step=2, sampling="log")

        bins = self.get_bins(hp)
        bin_width = bins[1] - bins[0]
        sigma = sig_ratio * bin_width
        return HLGaussian(self.base(), bins, sigma)
    
class HyperHLOneBin(HyperHL):

    def __init__(self, base, min_y, max_y, metrics=None):
        super().__init__(min_y, max_y, "HyperHL-OneBin", metrics)
        self.base = base

    def get_model(self, hp):
        bins = self.get_bins(hp)
        return HLOneBin(self.base(), bins)