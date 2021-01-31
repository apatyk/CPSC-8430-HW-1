### HW 1-1: Deep vs. Shallow
### Part 1: Simulate a function
### Adam Patyk
### CPSC 8430

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# model definitions
from common.DNN_Models import ShallowNetwork, ModerateNetwork, DeepNetwork
# Torch dataset for function data
from common.FunctionDataset import FunctionDataset

epochs = 10000
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

# train each model
training_loss = []
for i in range(len(models)):
  loss_arr = []
  models[i].zero_grad()

  print(f'Training model {i}:')

  for epoch in range(epochs):
    model_loss = models[i].train(training_loader)
    loss_arr.append(model_loss)
    if epoch % (epochs/10) == (epochs/10)-1: # print updates 10 times
      print(f'Epoch: {epoch+1}/{epochs} \tLoss: {model_loss:.6f}', flush=True)

  training_loss.append(loss_arr)
  torch.save(models[i].state_dict(), f'results/1/func_models/model{i}.pt')

# save results to .txt file
np.savetxt('results/1/func_models/training_loss.txt', np.array(training_loss))