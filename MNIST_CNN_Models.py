### HW 1-1: Deep vs. Shallow
### Part 2: Train on actual problem
### Adam Patyk
### CPSC 8430

import torch
import torch.nn as nn

# hyperparameters
epochs = 150
img_batch_size = 32
learning_rate = 0.01
momentum = 0.9

input_size = 1
output_size = 10
kernel_size = 3
pool_size = 2
conv_sizes = [16, 32]
fc_size = 20

class ShallowCNN(nn.Module):
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

    def forward(self, x):
        return self.model(x)

class ModerateCNN(nn.Module):
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

    def forward(self, x):
        return self.model(x)

class DeepCNN(nn.Module):
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

    def forward(self, x):
        return self.model(x)