### Adam Patyk
### CPSC 8430

from collections import OrderedDict
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
class _DNN(nn.Module):
  def __init__(self):
    super(_DNN, self).__init__()

  def train(self, data_loader):
    self.model.train()
    training_loss = 0.0
    
    for datum in data_loader:
      input, target = datum[0][0].reshape(-1), datum[0][1].reshape(-1)
      self.optimizer.zero_grad()
      output = self.model(input)
      loss = self.loss_function(output, target)
      loss.backward()
      self.optimizer.step()
      training_loss += loss.item()
    training_loss /= len(data_loader)

    return training_loss

  def test(self, data_loader):
    self.model.eval()
    testing_loss = 0.0
    correct = 0

    with torch.no_grad():
      for data, target in data_loader:
        output = self.model(data)
        loss = self.loss_function(output, target)
        testing_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
    total = len(data_loader.dataset)
    testing_loss /= total
    testing_acc = correct / total * 100

    return testing_acc, testing_loss

# Model 0
class ShallowNetwork(_DNN):
  def __init__(self):
    super(ShallowNetwork, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, shallow_hidden_size),
      nn.ReLU(),
      nn.Linear(shallow_hidden_size, output_size),
    )
    self.optimizer = optim.Adam(self.model.parameters())  # use adaptive learning rate over stochastic gradient descent
    self.loss_function = nn.MSELoss()                    # use mean-squared error loss function

  def forward(self, x):
    return self.model(x)

# Model 1
# OrderedDict needed for layer names for HW1-2a
class ModerateNetwork(_DNN):
  def __init__(self):
    super(ModerateNetwork, self).__init__()
    self.model = nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(input_size, mod_hidden_sizes[0])),
      ('activ1', nn.ReLU()),
      ('fc2', nn.Linear(mod_hidden_sizes[0], mod_hidden_sizes[1])),
      ('activ2', nn.ReLU()),
      ('fc3', nn.Linear(mod_hidden_sizes[1], mod_hidden_sizes[0])),
      ('activ3', nn.ReLU()),
      ('fc4', nn.Linear(mod_hidden_sizes[0], output_size)),
    ]))
    self.optimizer = optim.Adam(self.model.parameters())  # use adaptive learning rate over stochastic gradient descent
    self.loss_function = nn.MSELoss()                    # use mean-squared error loss function

  def forward(self, x):
    return self.model(x)

# Model 2
class DeepNetwork(_DNN):
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
    self.optimizer = optim.Adam(self.model.parameters())  # use adaptive learning rate over stochastic gradient descent
    self.loss_function = nn.MSELoss()                    # use mean-squared error loss function

  def forward(self, x):
    return self.model(x)