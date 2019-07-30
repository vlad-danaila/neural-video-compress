import loader
import util
from torch.utils.data import DataLoader
import numpy as np
import torch as t
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# PATH = '/content/drive/My Drive/mnist_test_seq.npy'
PATH = loader.DATASET_PATH

class MinimalCNN(t.nn.Module):
    def __init__(self):
        super(MinimalCNN, self).__init__()
        self.conv1 = t.nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 9, padding = 4)
    def forward(self, x):
        x = self.conv1(x)
        return x

class SimpleCNN(t.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = t.nn.Conv2d(in_channels=2, out_channels=6, kernel_size=5, padding=2)
        # self.conv1 = t.nn.Conv2d(3, 6, 5)
        self.pool = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(in_channels=6, out_channels=11, kernel_size=5, padding=2)
        # self.conv2 = t.nn.Conv2d(6, 16, 5)
        self.fc1 = t.nn.Linear(11 * 16 * 16, 64 * 64 * 5)
        self.fc2 = t.nn.Linear(64 * 64 * 5, 64 * 64 * 3)
        self.fc3 = t.nn.Linear(64 * 64 * 3, 64 * 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 11 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 1, 64, 64)

def train_minimal_cnn(path = loader.DATASET_PATH):
    minimalCNN = MinimalCNN()
    minimalCNN.to(device)
    dataloader = DataLoader(loader.MovingMNIST3Frames(path), batch_size=30, shuffle=False)
    criterion = t.nn.L1Loss()
    optimizer = optim.SGD(minimalCNN.parameters(), lr=0.1, momentum=0.9)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    running_loss = 0.0
    for j in range(5):
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
      scheduler.step()
    return minimalCNN

def plot_3_frames(model):
    frames = []
    dataset_3_slices = loader.MovingMNIST3Frames(PATH)
    x, y = dataset_3_slices.__getitem__(0)
    frame_1, frame_3 = x[0] * 255, x[1] * 255
    x = x.unsqueeze(0).to(device)
    frame_2 = (model(x)[0, 0] * 255).detach().cpu().numpy()
    [util.plot_grayscale(f) for f in [frame_1, frame_2, frame_3]]

if __name__ == '__main__':
    # model = train_minimal_cnn(PATH)
    model = SimpleCNN()
    plot_3_frames(model)