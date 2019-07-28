import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch as t
import util

# Dataset downloaded from http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
# Documentation at http://www.cs.toronto.edu/~nitish/unsupervised_video/
DATASET_PATH = '../../mnist_test_seq.npy'

def load_dataset():
    # Shape (20, 10000, 64, 64) sequence length, nr. of sequences, image sizes
    dataset = np.load(DATASET_PATH)
    # Shape (10000, 20, 64, 64) nr. of sequences, seq length, image sizes
    dataset = np.swapaxes(dataset, 0, 1)
    return dataset

def load_3_frames_slices():
    dataset = load_dataset()
    for frames in dataset:
        for i in range(len(frames) - 2):
            yield ((frames[i], frames[i + 2]), frames[i + 1])

def plot_animation_3_slices(limit = 2):
    frames = []
    for x, y in load_3_frames_slices():
        if limit == 0:
            break
        frame_1, frame_3 = x
        frame_2 = y
        frames += [frame_1, frame_2, frame_3]
        limit -= 1
    util.plot_animation(frames)

def plot_animation_3_slices_from_MovingMNIST3Frames(limit = 2):
    frames = []
    dataset_3_slices = MovingMNIST3Frames()
    for i in range(limit):
        x, y = dataset_3_slices.__getitem__(i)
        frame_1, frame_3 = x
        frame_2 = y
        frames += [frame_1, frame_2, frame_3]
    util.plot_animation(frames)

class MovingMNIST3Frames(Dataset):

    def __init__(self):
        self.dataset = load_dataset()
        self.count_movies = 0
        self.count_frames = 0
        self.nr_movies = self.dataset.shape[0]
        self.nr_slices = self.dataset.shape[1] - 2

    def __len__(self):
        return self.nr_movies * self.nr_slices

    def __getitem__(self, idx):
        i, j = self.count_movies, self.count_frames
        self.count_frames = (self.count_frames + 1) % self.nr_slices
        if self.count_frames == 0:
            self.count_movies = (self.count_movies + 1) % self.nr_movies
        x = np.stack([self.dataset[i, j], self.dataset[i, j + 2]], 0)
        y = np.expand_dims(self.dataset[i, j + 1], 0)
        x = t.from_numpy(x).float()
        y = t.from_numpy(y).float()
        return [x, y]

if __name__ == '__main__':
    dataset = load_dataset()

    # Plot a single image
    # util.plot_grayscale(dataset[1, 1])

    # Plot animation
    # util.plot_animation(dataset[2])

    # Plot animation of 3 slices
    # plot_animation_3_slices()

    # Plot 3 frames slices from MovingMNIST3Frames
    # plot_animation_3_slices_from_MovingMNIST3Frames(19)

    # Use pytorch dataset loader
    # dataloader = DataLoader(MovingMNIST3Frames(), batch_size=4, shuffle=True, num_workers=4)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched)