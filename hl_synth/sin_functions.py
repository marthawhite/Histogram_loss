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
from pathlib import Path


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the model
class RegressionVarDepth(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, depth=2):
        super(RegressionVarDepth, self).__init__()
        assert depth > 1
        
        # First layer (input to first hidden layer)
        layers = [nn.Linear(input_size, hidden_size)]
        layers.append(nn.LeakyReLU())
        
        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        
        # Output layer (last hidden layer to output)
        layers.append(nn.Linear(hidden_size, 1))
        
        # Combine the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HLVarDepth(nn.Module):
    def __init__(self, input_size=1, hidden_size=1024, depth=2, num_bins=100):
        super(HLVarDepth, self).__init__()
        assert depth > 1

        # First layer (input to first hidden layer)
        layers = [nn.Linear(input_size, hidden_size)]
        layers.append(nn.LeakyReLU())
        
        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        
        # Output layer (last hidden layer to output)
        layers.append(nn.Linear(hidden_size, num_bins))
        
        # Combine the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float, weights: torch.Tensor = None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        ).to(DEVICE)
        self.weights = weights
        if self.weights is None:
            self.weights = torch.ones(num_bins)
        self.weights = self.weights.to(DEVICE)
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weights)
    
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
    return f"Loss: {split[0]}, depth: {split[1]}, width: {split[2]}, lr: {split[3]}, Freq: {split[4]}, Offset: {split[5]}"


def main(model_name='HL-Gauss', depth=2, width=1024, lr=1e-1, Y_freq=1., Y_offset=0., hl_range=[-1.5, 1.5], seed=0, delete=False, task_idx=1):
    print('TASK IDX:', task_idx)
    task_name = f'{model_name}_{depth}_{width}_{lr}_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}_{seed}'
    # task_name = f'{model_name}_{lr}_{Y_freq}_{Y_offset}_{hl_range[0]}_{hl_range[1]}_{seed}'
    print(task_name)

    path = Path('results/ckpt')
    path.mkdir(parents=True, exist_ok=True)
    
    with h5py.File('results/sin_functions.hdf5', 'a') as f:
        if delete:
            if task_name in f:
                del f[task_name]
                print('deleted')
            else:
                print('not found')
            return

        # if task_name in f:
        #     print('done already')
        #     return

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Generate some data
    X = torch.linspace(-np.pi, np.pi, 501)[:-1].unsqueeze(1).to(DEVICE)
    Y = torch.sin(Y_freq*X) + Y_offset

    X_test = X + (X[1] - X[0])/2
    Y_test = torch.sin(Y_freq*X_test) + Y_offset

    if model_name == 'l2':
        model = RegressionVarDepth(hidden_size=width, depth=depth)
        criterion = nn.MSELoss()
    elif model_name == 'HL-Gauss':
        model = HLVarDepth(hidden_size=width, depth=depth)
        sigma = (hl_range[1] - hl_range[0])/100 * 2
        criterion = HLGaussLoss(hl_range[0], hl_range[1], 100, sigma)
        Y_probs = criterion.transform_to_probs(Y.squeeze())
    elif model_name == 'HL-Gauss-Balanced':
        model = HLVarDepth(hidden_size=width, depth=depth)
        sigma = (hl_range[1] - hl_range[0])/100 * 2
        class_weights = 1./(torch.histc(Y[:, 0], bins=100, min=hl_range[0], max=hl_range[1]))
        class_weights[class_weights == torch.inf] = 0.
        class_weights /= class_weights[class_weights.nonzero()].mean()
        criterion = HLGaussLoss(hl_range[0], hl_range[1], 100, sigma, weights=class_weights)
        Y_probs = criterion.transform_to_probs(Y.squeeze())
    else:
        raise NotImplementedError

    model = model.to(DEVICE)

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    # Training loop
    num_epochs = 1001

    log = {}
    log['train_mse'] = []
    log['test_mse'] = []

    for epoch in range(num_epochs):
        model.eval()

        Yhat = model(X)
        if model_name == 'HL-Gauss' or model_name == 'HL-Gauss-Balanced':
            Yhat = criterion.transform_from_probs(F.softmax(Yhat, dim=-1))
            mse = ((Y.squeeze() - Yhat)**2).mean()
        elif model_name == 'l2':
            mse = ((Y.squeeze() - Yhat.squeeze())**2).mean()
        else:
            raise NotImplementedError
        log['train_mse'].append(mse.item())

        Yhat_test = model(X_test)
        if model_name == 'HL-Gauss' or model_name == 'HL-Gauss-Balanced':
            Yhat_test = criterion.transform_from_probs(F.softmax(Yhat_test, dim=-1))
            mse_test = ((Y_test.squeeze() - Yhat_test)**2).mean()
        elif model_name == 'l2':
            mse_test = ((Y_test.squeeze() - Yhat_test.squeeze())**2).mean()
        else:
            raise NotImplementedError
        log['test_mse'].append(mse_test.item())
        # mse_test = 0.

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse:.4f}, Test MSE: {mse_test:.4f}, ")

        if torch.isnan(mse):
            print('nan occured')
            break
        
        model.train()

        outputs = model(X)
        if model_name == 'HL-Gauss' or model_name == 'HL-Gauss-Balanced':
            loss = criterion(outputs, Y_probs)
        elif model_name == 'l2':
            loss = criterion(outputs, Y)
        else:
            raise NotImplementedError
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    X_vis = torch.linspace(-np.pi, np.pi, 2001).unsqueeze(1).to(DEVICE)
    Yhat_vis = model(X_vis)
    if model_name == 'HL-Gauss':
        Yhat_vis = criterion.transform_from_probs(F.softmax(Yhat_vis, dim=-1)).unsqueeze(1)

    if seed == 0:
        plt.clf()
        plt.plot(X[:, 0].cpu(), Y[:, 0].cpu(), 'o')
        plt.plot(X_vis[:, 0].cpu(), Yhat_vis[:, 0].cpu().detach(), '-')
        plt.title(task_name_str(task_name))
        # plt.ylim(-1.1, 1.1)
        plt.savefig(f'results/vis_{task_name}.png', dpi=200)

    with h5py.File('results/sin_functions.hdf5', 'a') as f:
        if task_name in f:
            del f[task_name]
        f.create_group(task_name)
        for key, value in log.items():
            f.create_dataset(f"{task_name}/{key}", data=value)

    if seed == 0:
        torch.save(model.state_dict(), f'results/ckpt/{task_name}.pt')


if __name__ == '__main__':
    fire.Fire(main)