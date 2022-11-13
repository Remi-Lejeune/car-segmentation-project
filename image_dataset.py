import glob

import numpy as np
from torch.utils.data import Dataset


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

        x.append(list[0])
        y.append(list[3])

        x = np.array(x)
        y = np.array(y)

        return x, y


def files_name(path='carseg_data/clean_data/*.np[yz]'):
    return glob.glob(path)
