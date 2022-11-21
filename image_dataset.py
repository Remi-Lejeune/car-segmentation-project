import glob

import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch


class ImageDataset(Dataset):
    def __init__(self, files, size=None):

        self.files = files

        if size is not None and size < len(files):
            self.files = self.files[:size]

        if size is not None and size < len(self.files):
            self.files = self.files[:size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = []
        y = []
        list = np.load(self.files[idx]).astype(np.float32)

        # One-hot encoding of segmentation
        # seg = torch.tensor(list[3])
        # seg = one_hot(seg.to(torch.int64), num_classes=9)
        # seg = torch.permute(seg, (2, 0, 1))

        # Add image and segmentation to lists
        # x.append(list[i][0:3])
        # y.append(np.array(seg))

        x = np.array(list[:3])
        y = np.array(list[3])
        return x, y


def files_name(path='carseg_data/clean_data/*.np[yz]'):
    return glob.glob(path)
