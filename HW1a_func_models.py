### HW 1-1: Deep vs. Shallow
### Part 1: Simulate a function
### Adam Patyk
### CPSC 8430

import torch
import torch.nn as nn

# hyperparameters
epochs = 10000
input_size = 1
output_size = 1
shallow_hidden_size = 200
mod_hidden_sizes = [12, 22]
deep_hidden_size = 8

# Model 0
class ShallowNetwork(nn.Module):
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
class ModerateNetwork(nn.Module):
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
class DeepNetwork(nn.Module):
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