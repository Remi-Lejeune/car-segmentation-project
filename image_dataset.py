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
        seg = torch.tensor(list[3])
        seg = one_hot(seg.to(torch.int64), num_classes=9)
        seg = torch.permute(seg, (2, 0, 1))

        # Add image and segmentation to lists
        # x.append(list[i][0:3])
        # y.append(np.array(seg))

        x = np.array(list[:3])
        y = np.array(seg)
        return x, y


def files_name(path='carseg_data/clean_data/*.np[yz]'):
    return glob.glob(path)


def photos_file_name(path='carseg_data/carseg_raw_data/train/photo/*.jpg'):
    return glob.glob(path)


def get_photo_filenames(photos_file_names=photos_file_name()):
    clean_data_filenames = []
    for file_name in photos_file_names:
        image_file = file_name.split('\\')[1]
        image_name = image_file.split('.')[0]
        path = glob.glob("carseg_data/clean_data/"+image_name+".npy")
        clean_data_filenames.append(path[0])
    return clean_data_filenames


