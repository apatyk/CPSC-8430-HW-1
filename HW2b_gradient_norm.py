### HW 1-2: Optimization
### Part 2: Observe Gradient Norm During Training
### Adam Patyk
### CPSC 8430

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from common.DNN_Models import ModerateNetwork
from common.MNIST_CNN_Models import ModerateCNN
from common.FunctionDataset import FunctionDataset

def calc_grad_norm(model):
  grad_all = 0.0

  for p in model.parameters():
    grad = 0.0 
    if p.grad is not None:
      grad = (p.grad.cpu().data.numpy() ** 2).sum()
    grad_all += grad

  grad_norm = grad_all ** 0.5

  return grad_norm

DNN_epochs = 10000
CNN_epochs = 100
num_data_points = 100
range_min = 0
range_max = 1
img_batch_size = 32

DNN_model = ModerateNetwork()
CNN_model = ModerateCNN()

## ------------------------
## Create data for training
## ------------------------

# prepare data for e^x * sin^2(3*pi*x) function 
raw_func_data = FunctionDataset(num_data_points, range_min, range_max)
func_training_data = raw_func_data.standardize()
func_training_loader = DataLoader(func_training_data, batch_size=1, shuffle=True)

# prepare data for MNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
mnist_training_set = torchvision.datasets.MNIST(root='./data/', train=True, 
                                          download=True, transform=transform)
mnist_training_loader = torch.utils.data.DataLoader(mnist_training_set, 
                                          batch_size=img_batch_size,
                                          shuffle=True)

print('Data ready.')

## ---------------------------------
## Train models [est. time: 30 min] 
## ---------------------------------

overall_grad_norm = []
overall_loss = []
model_loss = []
model_grad_norm = []

DNN_model.zero_grad()
CNN_model.zero_grad()

start_time = time.time()

# train both models logging gradient norm and loss 
print('Training DNN model:')
for epoch in range(DNN_epochs):
    loss = DNN_model.train(func_training_loader)
    grad_norm = calc_grad_norm(DNN_model)
    model_loss.append(loss)
    model_grad_norm.append(grad_norm)
    if epoch % (DNN_epochs/10) == (DNN_epochs/10)-1: # print updates 10 times
      print(f'Epoch: {epoch+1}/{DNN_epochs}\tLoss: {loss:.6f}\tGrad norm: {grad_norm:.2f}', flush=True)

overall_loss.append(model_loss)
overall_grad_norm.append(model_grad_norm)

model_loss = []
model_grad_norm = []
print('Training CNN model:')
for epoch in range(CNN_epochs):
    loss = CNN_model.train(mnist_training_loader)
    grad_norm = calc_grad_norm(CNN_model)
    model_loss.append(loss)
    model_grad_norm.append(grad_norm)
    if epoch % (CNN_epochs/10) == (CNN_epochs/10)-1: # print updates 10 times
      print(f'Epoch: {epoch+1}/{CNN_epochs}\t\tLoss: {loss:.6f}\tGrad norm: {grad_norm:.2f}', flush=True)

total_time = (time.time() - start_time)
print(f'Training time: {total_time//60:.0f} min {total_time%60:.2f} s', flush=True)

overall_loss.append(model_loss)
overall_grad_norm.append(model_grad_norm)

## ---------------------------
## Plot loss and gradient norm
## ---------------------------

epochs = [DNN_epochs, CNN_epochs]

for i in range(2):
  fig, axs = plt.subplots(2)
  axs[0].plot(np.arange(0, epochs[i]), overall_grad_norm[i])
  axs[1].plot(np.arange(0, epochs[i]), overall_loss[i])
  plt.xlabel('Epochs')
  axs[0].set(ylabel='Grad')
  axs[1].set(ylabel='Loss')
  plt.savefig(f'results/2/gradient_norm{i}.pdf')
  plt.show()