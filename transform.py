import numpy as np
from scipy.special import erf
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

class HistNormTransform:
    
    
    def __init__(self, n_bins=100, sig_ratio=1.0, padding=0.1) -> None:
        self.n_bins = n_bins
        self.sig_ratio = sig_ratio
        self.pad_ratio = padding

    @classmethod
    def from_bins(cls, borders, sigma):
        obj = cls.__new__(cls)
        super(HistNormTransform, obj).__init__()
        obj.borders = borders
        obj.sigma = sigma
        obj.centers = (obj.borders[:, 1:] + obj.borders[:, :-1]) / 2.0
        return obj

    def get_centers(self):
        return self.centers

    def get_min_max(self, x):
        x_max = np.amax(x)
        x_min = np.amin(x)
        if self.pad_ratio >= 1:
            bins = self.n_bins
            self.n_bins += 2 * self.pad_ratio
            self.pad_ratio = self.pad_ratio * 1.0 / bins
        padding = (x_max - x_min) * self.pad_ratio
        x_max += padding
        x_min -= padding
        return x_min, x_max

    def fit(self, x):
        x_min, x_max = self.get_min_max(x)
        bin_size = (x_max - x_min) / self.n_bins
        self.sigma = self.sig_ratio * bin_size
        self.borders = np.linspace(x_min, x_max, self.n_bins + 1)
        self.centers = self.borders[:-1] + bin_size / 2.0

    def transform(self, x):
        border_targets = self.adjust_and_erf(self.borders[np.newaxis, :], x[:, np.newaxis], self.sigma)
        two_z = border_targets[:,-1] - border_targets[:,0]
        x_transformed = (border_targets[:, 1:] - border_targets[:, :-1]) / two_z[:,np.newaxis]
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def __str__(self) -> str:
        return f"n_bins: {self.n_bins}, Borders: {self.borders}"

    def adjust_and_erf(self, a, mu, sig):
        return erf((a - mu)/(np.sqrt(2.0)*sig))


def main():
    

    y_train = np.arange(0, 101, 1)

    ht = HistNormTransform()
    yt_train = ht.fit_transform(y_train)
    centers = ht.get_centers()
    y_recreated = np.dot(yt_train, centers)
    abs_err = np.abs(y_train - y_recreated)
    print(abs_err.mean())

    print(ht)
    


main()
