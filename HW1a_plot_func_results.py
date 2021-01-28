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
from DNN_Models import ShallowNetwork, ModerateNetwork, DeepNetwork
# Torch dataset for function data
from FunctionDataset import FunctionDataset

def inverse_standardize(data, mean, stddev):
    data = data * stddev + mean
    return data

epochs = 10000

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

# plot loss over epochs for each model 
for i in range(len(models)):
    plt.plot(np.arange(0, epochs), training_loss[i], color=colors[i], label=f'model{i}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(f'func_models/loss_comparison_model{i}.eps')
    plt.show()

# plot loss over epochs for all models
for i in range(len(models)):
    plt.plot(np.arange(0, epochs), training_loss[i], color=colors[i], label=f'model{i}')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig(f'func_models/loss_comparison.eps')
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