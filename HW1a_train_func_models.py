### HW 1-1: Deep vs. Shallow
### Part 1: Simulate a function
### Adam Patyk
### CPSC 8430

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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

def train(model, data_loader, epochs):
    optimizer = optim.Adam(model.parameters())  # use adaptive learning rate over stochastic gradient descent
    loss_func = nn.MSELoss()                    # use mean-squared error loss function
    model.zero_grad()
    training_loss = []

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for datum in data_loader:
            input, target = datum[0][0].reshape(-1), datum[0][1].reshape(-1)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        training_loss.append(epoch_loss)
        if epoch % (epochs/10) == (epochs/10)-1: # print updates 10 times
            print(f'Epoch: {epoch+1}/{epochs} \tLoss: {epoch_loss:.6f}', flush=True)
    
    total_time = (time.time() - start_time)
    print(f'Training time: {total_time//60:.0f} min {total_time%60:.2f} s')

    return training_loss

num_data_points = 100
range_min = 0
range_max = 1

## -----------------------------------------------
## Create models and validate number of parameters
## -----------------------------------------------

models = []
models.append(ShallowNetwork())
models.append(ModerateNetwork())
models.append(DeepNetwork())

for i in range(len(models)):
    num_params = sum(p.numel() for p in models[i].parameters())
    print(f'Model {i} parameters: {num_params}')

## ------------------------
## Create data for training
## ------------------------

# create data points from function e^x * sin^2(3*pi*x) over a range
raw_data = FunctionDataset(num_data_points, range_min, range_max)

# z-score standardize training data
training_data = raw_data.standardize()

# create data loader for batch training
training_loader = DataLoader(training_data, batch_size=1, shuffle=True)
print('Data ready.')

## ----------------------------------------------------
## Train models [est. time: ~1 hour with 10,000 epochs]
## ----------------------------------------------------

# train each model and save results to .txt file
training_loss = []
for i in range(len(models)):
    print(f'Training model {i}:')
    loss_arr = train(models[i], training_loader, epochs)
    training_loss.append(loss_arr)
    torch.save(models[i].state_dict(), f'func_models/model{i}.pt')
np.savetxt('func_models/training_loss.txt', np.array(training_loss))