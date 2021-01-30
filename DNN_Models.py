### HW 1-1: Deep vs. Shallow
### Part 1: Simulate a function
### Adam Patyk
### CPSC 8430

import torch
import torch.nn as nn
import torch.optim as optim

## --------------------------------------------------------
## DNN Model Definitions for Simulating Nonlinear Functions
## --------------------------------------------------------

# hyperparameters
input_size = 1
output_size = 1
shallow_hidden_size = 200
mod_hidden_sizes = [12, 22]
deep_hidden_size = 8

# parent class for DNN models with training method
class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()

  def train(self, data_loader):
    self.model.train()
    training_loss = 0.0
    optimizer = optim.Adam(self.model.parameters())  # use adaptive learning rate over stochastic gradient descent
    loss_function = nn.MSELoss()                    # use mean-squared error loss function
    
    for datum in data_loader:
      input, target = datum[0][0].reshape(-1), datum[0][1].reshape(-1)
      optimizer.zero_grad()
      output = self.model(input)
      loss = loss_function(output, target)
      loss.backward()
      optimizer.step()
      training_loss += loss.item()
    training_loss /= len(data_loader)

    return training_loss

# Model 0
class ShallowNetwork(DNN):
  def __init__(self):
    super(ShallowNetwork, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, shallow_hidden_size),
      nn.ReLU(),
      nn.Linear(shallow_hidden_size, output_size),
    )

  def forward(self, x):
    return self.model(x)

# Model 1
class ModerateNetwork(DNN):
  def __init__(self):
    super(ModerateNetwork, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, mod_hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(mod_hidden_sizes[0], mod_hidden_sizes[1]),
      nn.ReLU(),
      nn.Linear(mod_hidden_sizes[1], mod_hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(mod_hidden_sizes[0], output_size),
    )

  def forward(self, x):
    return self.model(x)

# Model 2
class DeepNetwork(DNN):
  def __init__(self):
    super(DeepNetwork, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, deep_hidden_size),
      nn.ReLU(),
      nn.Linear(deep_hidden_size, output_size),
    )

  def forward(self, x):
    return self.model(x)