### Adam Patyk
### CPSC 8430

import torch
import torch.nn as nn
import torch.optim as optim

## ---------------------------------------
## DNN Model Definitions for MNIST Dataset
## ---------------------------------------

# hyperparameters

mnist_input = 28 * 28
mnist_hidden_size = 256
mnist_output = 10

# parent class for DNN models with training method
class _DNN(nn.Module):
  def __init__(self):
    super(_DNN, self).__init__()

  def train(self, data_loader):
    self.model.train()
    training_loss = 0.0
    
    for datum in data_loader:
      input, target = datum[0].reshape(-1, mnist_input), datum[1]
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
    total = 0
    correct = 0

    with torch.no_grad():
      for datum in data_loader:
        input, target = datum[0].reshape(-1, mnist_input), datum[1]
        output = self.model(input)
        loss = self.loss_function(output, target)
        testing_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    testing_loss /= total
    testing_acc = correct / total * 100

    return testing_acc, testing_loss

class MNIST_DNN(_DNN):
  def __init__(self):
    super(MNIST_DNN, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(mnist_input, mnist_hidden_size),
      nn.ReLU(),
      nn.Linear(mnist_hidden_size, mnist_hidden_size),
      nn.ReLU(),
      nn.Linear(mnist_hidden_size, mnist_output)
    )
    self.optimizer = optim.Adam(self.model.parameters())  # use adaptive learning rate over stochastic gradient descent
    self.loss_function = nn.CrossEntropyLoss()   # cross entropy categorical loss function

  def forward(self, x):
    return self.model(x)