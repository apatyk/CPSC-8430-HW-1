### HW 1-1: Deep vs. Shallow
### Part 1: Simulate a function
### Adam Patyk
### CPSC 8430

import math
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# model definitions
from DNN_Models import epochs
from DNN_Models import ShallowNetwork, ModerateNetwork, DeepNetwork

# Torch dataset for function data (e^x * sin^2(3*pi*x))
class FunctionDataset(Dataset):
    def __init__(self, size, start, end):
        data = np.zeros((size, 2))
        data[:, 0] = np.arange(start, end, 1/size)
        data[:, 1] = np.exp(data[:, 0]) * np.sin(3 * math.pi * data[:, 0]) ** 2
        data = torch.from_numpy(data).float()
        self.x = data
        self.mean = torch.mean(data)
        self.stddev = torch.std(data)
    
    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __mean__(self):
        return self.mean

    def __stddev__(self):
        return self.stddev

    def standardize(self):
        data = (self.x - self.mean) / self.stddev
        return data


def inverse_standardize(data, mean, stddev):
    data = data * stddev + mean
    return data

## ------------
## Plot results
## ------------

colors = ['tab:orange', 'tab:green', 'tab:purple']

# load model parameters from previous training
models = []
models.append(ShallowNetwork())
models.append(ModerateNetwork())
models.append(DeepNetwork())

for i in range(len(models)):
  saved_model = torch.load(f'func_models/model{i}.pt')
  models[i].load_state_dict(saved_model)

# load data from previous training
training_loss = np.loadtxt('func_models/training_loss.txt')

# create 100 data points from function e^x * sin^2(3*pi*x) [0, 1]
raw_data = FunctionDataset(100, 0, 1)

# z-score standardize training data
testing_data = raw_data.standardize()

# plot loss over epochs
for i in range(len(models)):
    plt.plot(np.arange(0, epochs), training_loss[i], color=colors[i], label=f'model{i}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(f'func_models/loss_comparison_model{i}.eps')
    plt.show()

# plot values from final model vs. ground truth function
plt.plot(raw_data[:, 0], raw_data[:, 1], label=r'$e^xsin(\pi x)$')
for i in range(len(models)):
    func_vals = []
    for datum in testing_data:
        input = datum[0].reshape(-1)
        output = models[i](input)
        output = inverse_standardize(output, raw_data.mean, raw_data.stddev)
        func_vals.append(output)
    plt.plot(raw_data[:, 0], func_vals, color=colors[i],label=f'model{i}')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('func_models/output_comparison.eps')
plt.show()