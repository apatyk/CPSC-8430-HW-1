### HW 1-1: Deep vs. Shallow
### Part 2: Train on actual problem
### Adam Patyk
### CPSC 8430

import numpy as np
import torch
import matplotlib.pyplot as plt
# model definitions
from MNIST_CNN_Models import ShallowCNN, ModerateCNN, DeepCNN

epochs = 150
colors = ['tab:orange', 'tab:green', 'tab:purple']

## ------------
## Plot results
## ------------

# load model parameters from previous training
models = []
models.append(ShallowCNN())
models.append(ModerateCNN())
models.append(DeepCNN())

for i in range(len(models)):
  models[i] = torch.load(f'mnist_models/model{i}.pt')

# load data from previous training
training_loss = np.loadtxt('mnist_models/training_loss.txt')
training_acc = np.loadtxt('mnist_models/training_acc.txt')

# plot training loss over epochs
for i in range(len(models)):
    plt.plot(np.arange(0, epochs), training_loss[i], color=colors[i], label=f'model{i}')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig('mnist_models/training_loss_comparison.eps')
plt.show()

# plot training accuracy over epochs
for i in range(len(models)):
    plt.plot(np.arange(0, epochs), training_acc[i], color=colors[i], label=f'model{i}')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('mnist_models/training_acc_comparison.eps')
plt.show()