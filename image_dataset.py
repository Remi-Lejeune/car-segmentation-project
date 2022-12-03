import glob
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
import os
from augmentation import *


def get_transformed_items(input, output):
    prob = np.random.randint(low=0, high=5)
    if prob == 0:
        return input, output
    if prob == 1:
        return rotate_image(input, output)
    if prob == 2:
        return similarity_transform_image(input, output)
    if prob == 3:
        return zoom_im(input, output)
    if prob == 4:
        return flip_im(input, output)


def get_output_masks(output):
    mask = torch.tensor(output)
    masks = one_hot(mask.to(torch.int64), num_classes=9)
    masks = torch.permute(masks, (2, 0, 1))
    return np.array(masks)


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
        image = np.load(self.files[idx]).astype(np.float32)

        gray_scaled_image = rgb2gray(image[:3])

        input, output = get_transformed_items(gray_scaled_image, image[3])

        return input, get_output_masks(output)


def files_name(path='carseg_data/clean_data/*.np[yz]'):
    return glob.glob(path)


# function that outputs all files with a number and numpy extension
def get_test_files():
    exclude = ["carseg_data/clean_data/0_a.npy", "carseg_data/clean_data/1_a.npy", "carseg_data/clean_data/2_a.npy",
               "carseg_data/clean_data/3_a.npy", "carseg_data/clean_data/5_a.npy", "carseg_data/clean_data/6_a.npy",
               "carseg_data/clean_data/10_a.npy", "carseg_data/clean_data/11_a.npy", "carseg_data/clean_data/12_a.npy",
               "carseg_data/clean_data/19_a.npy", "carseg_data/clean_data/20_a.npy",
               "carseg_data/clean_data/21_a.npy", "carseg_data/clean_data/22_a.npy", "carseg_data/clean_data/24_a.npy",
               "carseg_data/clean_data/26_a.npy", "carseg_data/clean_data/28_a.npy", "carseg_data/clean_data/29_a.npy",
               "carseg_data/clean_data/32_a.npy", "carseg_data/clean_data/33_a.npy", "carseg_data/clean_data/35_a.npy",
               "carseg_data/clean_data/36_a.npy", "carseg_data/clean_data/39_a.npy", "carseg_data/clean_data/40_a.npy",
               "carseg_data/clean_data/43_a.npy", "carseg_data/clean_data/45_a.npy", "carseg_data/clean_data/46_a.npy",
               "carseg_data/clean_data/47_a.npy", "carseg_data/clean_data/50_a.npy", "carseg_data/clean_data/51_a.npy",
               "carseg_data/clean_data/52_a.npy"
               ]

    include = (exclude)

    return include


def get_clean_files(path='carseg_data/clean_data/'):
    files = get_files(path)
    files_a = get_test_files()
    return np.setdiff1d(files, files_a)


def get_files(path='carseg_data/clean_data/'):
    return glob.glob(path + "[0-9]*.np[yz]")


def photos_file_name(path='carseg_data/carseg_raw_data/train/photo/*.jpg'):
    list = [os.path.normpath(i) for i in glob.glob(path)]
    return list


def get_npy_filenames(photos_file_names):
    clean_data_filenames = []
    for file_name in photos_file_names:
        image_file = file_name.split("/")[-1]
        image_name = image_file.split('.')[0]
        path_list = glob.glob("carseg_data/clean_data/" + image_name + ".npy")
        clean_data_filenames.append(path_list[0])
    return clean_data_filenames


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, rgb.shape[1], rgb.shape[2])
