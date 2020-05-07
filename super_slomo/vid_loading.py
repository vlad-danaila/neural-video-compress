import sys
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import os
import vid_util


class VidDataset(t.utils.data.Dataset):

    # time_span in milliseconds
    def __init__(self, dir, time_span, transform):
        self.dir = dir
        self.time_span = time_span
        self.transform = transform

    def __getitem__(self, i):
        pass

    def __len__(self):
        return os.listdir(dir)

if __name__ == '__main__':
    print(vid_util.TRAIN_PATH)
    train_ds = VidDataset(vid_util.TRAIN_PATH, 100, )
