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


class RegressionVarDepth(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, depth=3):
        super(RegressionVarDepth, self).__init__()
        
        # First layer (input to first hidden layer)
        layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]
        
        # Hidden layers (depth - 1 hidden layers)
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        
        # Output layer (last hidden layer to output)
        layers.append(nn.Linear(hidden_size, 1))
        
        # Combine the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HLVarDepth(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, depth=3, num_bins=100):
        super(HLVarDepth, self).__init__()
        
        # First layer (input to first hidden layer)
        layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]
        
        # Hidden layers (depth - 1 hidden layers)
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        
        # Output layer (last hidden layer to output)
        layers.append(nn.Linear(hidden_size, num_bins))
        
        # Combine the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


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
    return f"Loss: {split[0]}, depth: {split[1]}, lr: {split[2]}, Freq: {split[3]}, Offset: {split[4]}"


def main(model_name='l2', depth=3, lr=1e-3, Y_freq=20., Y_offset=0., hl_range=[-1.5, 1.5], seed=0, delete=False):
    task_name = f'{model_name}_{depth}_{lr}_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}_{seed}'
    print(task_name)
    assert seed == 0

    X = torch.linspace(-np.pi, np.pi, 501)[:-1].unsqueeze(1).to(DEVICE)
    Y = torch.sin(Y_freq*X) + Y_offset

    if model_name == 'l2':
        model = RegressionVarDepth(depth=depth)
        criterion = nn.MSELoss()
    elif model_name == 'HL-Gauss':
        model = HLVarDepth(depth=depth)
        sigma = (hl_range[1] - hl_range[0])/100 * 2
        criterion = HLGaussLoss(hl_range[0], hl_range[1], 100, sigma)
    else:
        raise NotImplementedError

    model = model.to(DEVICE)
    model.load_state_dict(torch.load((f'results/ckpt/{task_name}.pt'), map_location=torch.device('cpu')))

    model.eval()
    X_vis = torch.linspace(-np.pi, np.pi, 2001).unsqueeze(1).to(DEVICE)
    Yhat = model(X)
    Yhat_vis = model(X_vis)
    if model_name == 'HL-Gauss':
        Yhat = criterion.transform_from_probs(F.softmax(Yhat, dim=-1)).unsqueeze(1)
        Yhat_vis = criterion.transform_from_probs(F.softmax(Yhat_vis, dim=-1)).unsqueeze(1)

    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(X[:, 0].cpu(), Y[:, 0].cpu(), 'o')
    plt.plot(X_vis[:, 0].cpu(), Yhat_vis[:, 0].cpu().detach(), '-', linewidth=2)
    # plt.title(task_name_str(task_name))
    plt.tight_layout()
    plt.savefig(f'results/clean_vis_{task_name}.png', dpi=200)


if __name__ == '__main__':
    fire.Fire(main)