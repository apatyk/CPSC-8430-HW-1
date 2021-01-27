### HW 1-1: Deep vs. Shallow
### Part 2: Train on actual problem
### Adam Patyk
### CPSC 8430

import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# model definitions and hyperparameters
from HW1b_mnist_models import epochs, img_batch_size, learning_rate, momentum
from HW1b_mnist_models import ShallowCNN, ModerateCNN, DeepCNN

def train(model, data_loader):   
    model.train() 
    training_loss = 0.0

    for data, target in data_loader:
      optimizer.zero_grad()
      output = model(data)
      loss = loss_func(output, target)
      loss.backward()
      optimizer.step()
      training_loss += loss.item()
    training_loss /= len(data_loader)

    return training_loss

def test(model, data_loader):
    model.eval()
    testing_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = loss_func(output, target)
            testing_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    total = len(data_loader.dataset)
    testing_loss /= total
    testing_acc = correct / total * 100
    #print(f'Accuracy: {correct}/{total} ({testing_acc:.2f}%)\tLoss: {testing_loss:.6f}', flush=True)

    return testing_acc

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
    optimizer = optim.SGD(models[i].parameters(), lr=learning_rate, momentum=momentum) # stochastic gradient descent
    loss_func = nn.CrossEntropyLoss()   # cross entropy categorical loss function

    print(f'Training model {i}:')
    start_time = time.time()

    for epoch in range(epochs):
      # train
      model_loss = train(models[i], training_loader)
      loss_arr.append(model_loss)
      # test
      model_acc = test(models[i], training_loader)
      acc_arr.append(model_acc)
      # print updates 10 times
      if epoch % (epochs/10) == (epochs/10)-1: 
        print(f'Epoch: {epoch+1}/{epochs} \tLoss: {model_loss:.6f} \tAccuracy: {model_acc:.2f}', flush=True)
    
    training_loss.append(loss_arr)
    training_acc.append(acc_arr)
    total_time = (time.time() - start_time)
    print(f'Training time: {total_time//60:.0f} min {total_time%60:.2f} s', flush=True)
    torch.save(models[i], f'mnist_models/model{i}.pt')

# save results to .txt files
np.savetxt('mnist_models/training_loss.txt', np.array(training_loss))
np.savetxt('mnist_models/training_acc.txt', np.array(training_acc))

## ----------------------------------------------------------
## Test models
## ----------------------------------------------------------

for i in range(len(models)):
  print(f'Testing model {i}:')
  model_acc = test(models[i], testing_loader)
  print(f'Accuracy: {model_acc:.2f}', flush=True)