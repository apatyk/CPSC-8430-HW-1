### Adam Patyk
### CPSC 8430

import torch
import torch.nn as nn
import torch.optim as optim

## ---------------------------------------
## CNN Model Definitions for MNIST Dataset
## ---------------------------------------

# hyperparameters
learning_rate = 0.01
momentum = 0.9

input_size = 1
output_size = 10
kernel_size = 3
pool_size = 2
conv_sizes = [16, 32]
fc_size = 20

# parent class for CNN models with training, testing methods
class _CNN(nn.Module):
  def __init__(self):
    super(_CNN, self).__init__()

  def train(self, data_loader):   
    self.model.train() 
    training_loss = 0.0

    for data, target in data_loader:
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.loss_function(output, target)
      loss.backward()
      self.optimizer.step()
      training_loss += loss.item()
    training_loss /= len(data_loader)

    return training_loss

  def test(self, data_loader):
    self.model.eval()
    testing_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():
      for data, target in data_loader:
        output = self.model(data)
        loss = self.loss_function(output, target)
        testing_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    testing_loss /= total
    testing_acc = correct / total * 100

    return testing_acc, testing_loss

class ShallowCNN(_CNN):
  def __init__(self):
    super(ShallowCNN, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(input_size, conv_sizes[0], kernel_size),
      nn.MaxPool2d(pool_size, pool_size),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(2704, fc_size),
      nn.ReLU(),
      nn.Linear(fc_size, output_size)
    )
    self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum) # stochastic gradient descent
    self.loss_function = nn.CrossEntropyLoss()   # cross entropy categorical loss function

  def forward(self, x):
    return self.model(x)

class ModerateCNN(_CNN):
  def __init__(self):
    super(ModerateCNN, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(input_size, conv_sizes[0], kernel_size),
      nn.MaxPool2d(pool_size, pool_size),
      nn.ReLU(),
      nn.Conv2d(conv_sizes[0], conv_sizes[1], kernel_size),
      nn.MaxPool2d(pool_size, pool_size),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(800, fc_size),
      nn.ReLU(),
      nn.Linear(fc_size, output_size)
    )
    self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum) # stochastic gradient descent
    self.loss_function = nn.CrossEntropyLoss()   # cross entropy categorical loss function


  def forward(self, x):
    return self.model(x)

class DeepCNN(_CNN):
  def __init__(self):
    super(DeepCNN, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(input_size, conv_sizes[0], kernel_size),
      nn.MaxPool2d(pool_size, pool_size),
      nn.ReLU(),
      nn.Conv2d(conv_sizes[0], conv_sizes[1], kernel_size),
      nn.MaxPool2d(pool_size, pool_size),
      nn.ReLU(),
      nn.Conv2d(conv_sizes[1], conv_sizes[1], kernel_size),
      nn.MaxPool2d(pool_size, pool_size),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(32, fc_size),
      nn.ReLU(),
      nn.Linear(fc_size, output_size)
    )
    self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum) # stochastic gradient descent
    self.loss_function = nn.CrossEntropyLoss()   # cross entropy categorical loss function


  def forward(self, x):
    return self.model(x)