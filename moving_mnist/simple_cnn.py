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

def train_minimal_cnn(path = loader.DATASET_PATH):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    minimalCNN = MinimalCNN()
    minimalCNN.to(device)
    dataloader = DataLoader(loader.MovingMNIST3Frames(path), batch_size=50, shuffle=False)
    criterion = t.nn.L1Loss()
    optimizer = optim.SGD(minimalCNN.parameters(), lr=0.01, momentum=0.9)
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        # x shape = 50 x 2 x 64 x 64
        # y shape = 50 x 1 x 64 x 64
        x, y_real = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = minimalCNN(x)
        loss = criterion(y_pred, y_real)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:  # print every 100 mini-batches
            print('[%d] loss: %.3f' % (i, running_loss / 10))
            running_loss = 0.0

if __name__ == '__main__':
    train_minimal_cnn()
