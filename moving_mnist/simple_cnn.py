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
        self.conv1 = t.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        return x


class SimpleCNN(t.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = t.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, padding=2)
        self.pool = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.fc1 = t.nn.Linear(16 * 16 * 16, 64 * 64)
        self.fc1 = t.nn.Linear(16 * 16 * 16, 64 * 64 * 3)
        self.fc2 = t.nn.Linear(64 * 64 * 3, 64 * 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FullyCNN(t.nn.Module):
    def __init__(self):
        super(FullyCNN, self).__init__()
        self.conv1 = t.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, padding=2)
        self.pool = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.deconv1 = t.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.deconv2 = t.nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x


def train_cnn(cnn, path=loader.DATASET_PATH):
    dataloader = DataLoader(loader.MovingMNIST3Frames(path), batch_size=30, shuffle=False)
    criterion = t.nn.L1Loss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    running_loss = 0.0
    for j in range(1):
        print('EPOCH ', j)
        for i, data in enumerate(dataloader):
            # x shape = 50 x 2 x 64 x 64
            # y shape = 50 x 1 x 64 x 64
            x, y_real = data[0].to(device), data[1].to(device)

            # Uncomment this if using fully connected layers at the end
            # y_real = y_real.view(-1, 64 * 64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = cnn(x)
            loss = criterion(y_pred, y_real)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # print every 100 mini-batches
                print('[%d] loss: %.3f' % (i, running_loss / 100))
                running_loss = 0.0
        scheduler.step()
    return cnn


def plot_3_frames_with_fully_connected_layers(model):
    frames = []
    dataset_3_slices = loader.MovingMNIST3Frames(PATH)
    x, y = dataset_3_slices.__getitem__(0)
    frame_1, frame_3 = x[0] * 255, x[1] * 255
    x = x.unsqueeze(0).to(device)
    frame_2 = model(x)
    frame_2 = frame_2.view(64, 64)
    frame_2 = frame_2 * 255
    frame_2 = frame_2.detach().cpu().numpy()
    [util.plot_grayscale(f) for f in [frame_1, frame_2, frame_3]]


def plot_3_frames_with_fully_convolutional(model):
    frames = []
    dataset_3_slices = loader.MovingMNIST3Frames(PATH)
    x, y = dataset_3_slices.__getitem__(0)
    frame_1, frame_3 = x[0] * 255, x[1] * 255
    x = x.unsqueeze(0).to(device)
    frame_2 = model(x)[0, 0]
    frame_2 = frame_2 * 255
    frame_2 = frame_2.detach().cpu().numpy()
    [util.plot_grayscale(f) for f in [frame_1, frame_2, frame_3]]


if __name__ == '__main__':
    model = FullyCNN().to(device)
    model = train_cnn(model, PATH)
    plot_3_frames_with_fully_convolutional(model)