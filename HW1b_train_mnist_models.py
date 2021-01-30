### HW 1-1: Deep vs. Shallow
### Part 2: Train on actual problem
### Adam Patyk
### CPSC 8430

import time
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
# model definitions and hyperparameters
from common.MNIST_CNN_Models import ShallowCNN, ModerateCNN, DeepCNN

epochs = 150
img_batch_size = 32

## -------------
## Create models
## -------------

models = []
models.append(ShallowCNN())
models.append(ModerateCNN())
models.append(DeepCNN())

print('Models ready.')

## ------------------------------------------------
## Load data, normalized (mean = 0.5, stddev = 0.5)
## ------------------------------------------------

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

training_set = torchvision.datasets.MNIST(root='./data/', train=True, 
                                            download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=img_batch_size,
                                          shuffle=True)

testing_set = torchvision.datasets.MNIST(root='./data/', train=False,
                                            download=True, transform=transform)
testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=img_batch_size,
                                         shuffle=False)

print('Data ready.')

## ----------------------------------------------------------
## Train & test models [est. time: 1.5 hours with 150 epochs]
## ----------------------------------------------------------

training_loss = []
training_acc = []

for i in range(len(models)):
  loss_arr = []
  acc_arr = []

  models[i].zero_grad()

  print(f'Training model {i}:')
  start_time = time.time()

  for epoch in range(epochs):
    # train
    model_loss = models[i].train(training_loader)
    loss_arr.append(model_loss)
    # test
    model_acc = models[i].test(training_loader)
    acc_arr.append(model_acc)
    # print updates 10 times
    if epoch % (epochs/10) == (epochs/10)-1: 
      print(f'Epoch: {epoch+1}/{epochs} \tLoss: {model_loss:.6f} \tAccuracy: {model_acc:.2f}', flush=True)
  
  training_loss.append(loss_arr)
  training_acc.append(acc_arr)
  total_time = (time.time() - start_time)
  print(f'Training time: {total_time//60:.0f} min {total_time%60:.2f} s', flush=True)
  torch.save(models[i].state_dict(), f'results/1/mnist_models/model{i}.pt')

# save results to .txt files
np.savetxt('results/1/mnist_models/training_loss.txt', np.array(training_loss))
np.savetxt('results/1/mnist_models/training_acc.txt', np.array(training_acc))

## ----------------------------------------------------------
## Test models
## ----------------------------------------------------------

for i in range(len(models)):
  print(f'Testing model {i}:')
  model_acc = models[i].test(testing_loader)
  print(f'Accuracy: {model_acc:.2f}', flush=True)