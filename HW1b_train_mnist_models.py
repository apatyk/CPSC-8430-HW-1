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

def train(model, data_loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # stochastic gradient descent
    loss_func = nn.CrossEntropyLoss()   # Cross entropy categorical loss function
    model.zero_grad()
    model.train()

    overall_loss = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        overall_loss.append(epoch_loss)
        if epoch % (epochs/10) == (epochs/10)-1: # print updates 10 times
            print(f'Epoch: {epoch+1}/{epochs} \tLoss: {epoch_loss:.6f}', flush=True)
    total_time = (time.time() - start_time)
    print(f'Training time: {total_time//60:.0f} min {total_time%60:.2f} s', flush=True)

    return overall_loss

def test(model, data_loader):
    loss_func = nn.CrossEntropyLoss()   # Cross entropy categorical loss function

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
    print(f'Accuracy: {correct}/{total} ({testing_acc:.2f}%)\tLoss: {testing_loss:.6f}', flush=True)

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

## ---------------------------------------------------
## Train models [est. time: 1.5 hours with 150 epochs]
## ---------------------------------------------------

training_loss = []

# train each model and save results to .txt file
for i in range(len(models)):
    print(f'Training model {i}:')
    loss_arr = train(models[i], training_loader, epochs)
    training_loss.append(loss_arr)
    torch.save(models[i], f'mnist_models/model{i}.pt')

np.savetxt('mnist_models/training_loss.txt', np.array(training_loss))

## -----------
## Test models
## -----------

# test each model and save results to .txt file
testing_loss = []
for i in range(len(models)):
    print(f'Testing model {i}:')
    loss_arr = test(models[i], testing_loader)
    testing_loss.append(loss_arr)