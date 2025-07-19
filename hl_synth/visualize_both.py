import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.special
import fire
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sin_functions import RegressionVarDepth, HLVarDepth
import matplotlib


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the model
class RegressionModel(nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.h1 = nn.Linear(1, width)
        self.h2 = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)
    
    def forward(self, x):
        x = self.h1(x)
        x = F.leaky_relu(x)
        x = self.h2(x)
        x = F.leaky_relu(x)
        return self.out(x)


class HLGaussModel(nn.Module):
    def __init__(self, width=1024, num_bins=100):
        super().__init__()
        self.h1 = nn.Linear(1, width)
        self.h2 = nn.Linear(width, width)
        self.out = nn.Linear(width, num_bins)
    
    def forward(self, x):
        x = self.h1(x)
        x = F.leaky_relu(x)
        x = self.h2(x)
        x = F.leaky_relu(x)
        return self.out(x)


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        ).to(DEVICE)
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0).to(DEVICE)) * self.sigma)
            )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)


def task_name_str(task_name):
    split = task_name.split('_')
    return f"Loss: {split[0]}, lr: {split[1]}, Freq: {split[2]}, Offset: {split[3]}"


def main(Y_freq=10, Y_offset=0, hl_range=[-1.5, 1.5], depth=3, width=1024):

    X = torch.linspace(-np.pi, np.pi, 501)[:-1].unsqueeze(1).to(DEVICE)
    Y = torch.sin(Y_freq*X) + Y_offset
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(X[:, 0].cpu(), Y[:, 0].cpu(), '-', linewidth=2, color="black", ms=5)

    methods = [('l2', 1e-3), ('HL-Gauss', 1e-2)]
    for model_name, lr in methods:
        task_name = f'{model_name}_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}_{0}'
        print(task_name)
        if model_name == 'l2':
            model = RegressionVarDepth(depth=depth, hidden_size=width)
            criterion = nn.MSELoss()
            # color = hsv_to_rgb((.03, 1., 0.8))
            color = matplotlib.colormaps["tab10"].colors[0]
            model_label = "$\\ell_2$"
        elif model_name == 'HL-Gauss':
            model = HLVarDepth(depth=depth, hidden_size=width)
            sigma = (hl_range[1] - hl_range[0])/100 * 2
            criterion = HLGaussLoss(hl_range[0], hl_range[1], 100, sigma)
            # color = hsv_to_rgb((.3, 1., 0.8))
            color = matplotlib.colormaps["tab10"].colors[1]
            model_label = model_name
        else:
            raise NotImplementedError

        model = model.to(DEVICE)
        model.load_state_dict(torch.load((f'results/ckpt/{task_name}.pt'), map_location=torch.device('cpu')))

        model.eval()
        X_vis = torch.linspace(-np.pi, np.pi, 2001).unsqueeze(1).to(DEVICE)
        Yhat_vis = model(X_vis)
        if model_name == 'HL-Gauss':
            Yhat_vis = criterion.transform_from_probs(F.softmax(Yhat_vis, dim=-1)).unsqueeze(1)

        plt.plot(X_vis[:, 0].cpu(), Yhat_vis[:, 0].cpu().detach(), '-', linewidth=3, color=color, label=model_label)
    # plt.title(task_name_str(task_name))
    plt.xticks([-np.pi, 0, np.pi], labels=['$-\\pi$', '$0$', '$-\\pi$'])
    # plt.legend(fontsize='small', framealpha=0.9)
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'results/clean_vis_{Y_freq}_{Y_offset}.png', dpi=200)


if __name__ == '__main__':
    fire.Fire(main)