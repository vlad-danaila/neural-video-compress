import loader
from torch.utils.data import DataLoader

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn.functional as F

dataloader = DataLoader(loader.MovingMNIST3Frames(), batch_size=50, shuffle=False)
for i, data in enumerate(dataloader):
    x_init, x_fin, y = data[0][0], data[0][1], data[1]
    print(i)
