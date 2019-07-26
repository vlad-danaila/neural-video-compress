import numpy as np
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
        frames.append(frame_1)
        frames.append(frame_2)
        frames.append(frame_3)
        limit -= 1
    util.plot_animation(frames)

if __name__ == '__main__':
    dataset = load_dataset()

    # Plot a single image
    # util.plot_grayscale(dataset[1, 1])

    # Plot animation
    # util.plot_animation(dataset[2])

    # Plot animation of 3 slices
    plot_animation_3_slices()