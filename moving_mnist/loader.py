import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Dataset downloaded from http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
# Documentation at http://www.cs.toronto.edu/~nitish/unsupervised_video/
from torch.utils import data

DATASET_PATH = '../../mnist_test_seq.npy'

# Shape (20, 10000, 64, 64) sequence length, nr. of sequences, image sizes
dataset = np.load(DATASET_PATH)

# Shape (10000, 20, 64, 64) nr. of sequences, seq length, image sizes
dataset = np.swapaxes(dataset, 0, 1)

imgplot = plt.imshow(dataset[0][0], cmap='gray', vmin=0, vmax=255)
plt.show()