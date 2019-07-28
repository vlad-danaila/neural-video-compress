import loader
from torch.utils.data import DataLoader

import numpy as np
import torch as t
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class MinimalCNN(t.nn.Module):
    def __init__(self):
        super(MinimalCNN, self).__init__()
        self.conv1 = t.nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 9, padding = 4)
    def forward(self, x):
        x = self.conv1(x)
        return x

minimalCNN = MinimalCNN()
dataloader = DataLoader(loader.MovingMNIST3Frames(), batch_size=50, shuffle=False)
criterion = t.nn.L1Loss()
optimizer = optim.SGD(minimalCNN.parameters(), lr=0.01, momentum=0.9)
running_loss = 0.0
for i, data in enumerate(dataloader):
    # x shape = 50 x 2 x 64 x 64
    # y shape = 50 x 1 x 64 x 64
    x, y_real = data[0], data[1]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    y_pred = minimalCNN(x)
    loss = criterion(y_pred, y_real)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 10 == 0:  # print every 100 mini-batches
        print('[%d] loss: %.3f' %
              (i, running_loss / 10))
        running_loss = 0.0


