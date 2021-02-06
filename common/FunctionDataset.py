### Adam Patyk
### CPSC 8430

import math
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Torch dataset for function data (e^x * sin^2(3*pi*x))
class FunctionDataset(Dataset):
    def __init__(self, size, start, end):
        data = np.zeros((size, 2))
        data[:, 0] = np.arange(start, end, 1/size)
        data[:, 1] = np.exp(data[:, 0]) * np.sin(3 * math.pi * data[:, 0]) ** 2
        data = torch.from_numpy(data).float()
        self.x = data
        self.mean = torch.mean(data)
        self.stddev = torch.std(data)
    
    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __mean__(self):
        return self.mean

    def __stddev__(self):
        return self.stddev

    def standardize(self):
        data = (self.x - self.mean) / self.stddev
        return data