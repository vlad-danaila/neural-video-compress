import loader
from torch.utils.data import DataLoader

import numpy as np
import torch as t
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn.functional as F


class SimpleCNN(t.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = t.nn.Conv2d(2, 6, 5)
        self.pool = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

simpleCNN = SimpleCNN()
dataloader = DataLoader(loader.MovingMNIST3Frames(), batch_size=50, shuffle=False)
for i, data in enumerate(dataloader):
    # x shape = 50 x 2 x 64 x 64
    # y shape = 50 x 1 x 64 x 64
    x, y_real = data[0], data[1]
    y_pred = simpleCNN(x)
    print(y_pred)
