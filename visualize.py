import matplotlib.pyplot as plt
from torch.nn.functional import one_hot

from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from Unet_pp_v2 import Unet_pp
import torch
from skimage import filters


def load(checkpoint, random=True, size=1, path=None):
    if path is not None:
        files = files_name(path)
    else:
        files = files_name()

    if random:
        np.random.shuffle(files)

    dataset = ImageDataset(files, size=size)
    dataloader = DataLoader(dataset, batch_size=size, shuffle=random)

    model = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint)
    #
    # # disable randomness, dropout, etc...
    model.eval()

    x, y = next(iter(dataloader))

    # predict with the model
    y_hat = model(x)

    y_hat = y_hat.detach().numpy()
    y = y.detach().numpy()
    x = x.numpy()
    x = x.transpose(0, 2, 3, 1)

    return x, y, y_hat


def plot_difference(y, y_hat):
    y = convert(y)

    fig, axs = plt.subplots(nrows=9, ncols=3, figsize=(3 * 9, 9 * 9))
    axs[0, 0].set_title("y")
    axs[0, 1].set_title("y_hat")
    axs[0, 2].set_title("difference")

    for i in range(9):
        axs[i, 0].imshow(y[i], vmin=0, vmax=1)
        axs[i, 1].imshow(y_hat[i], vmin=0, vmax=1)
        axs[i, 2].imshow(np.abs(y_hat[i] - y[i]), vmin=0, vmax=1)

    fig.tight_layout()
    plt.show()


def apply(x, masks):
    image = x.copy()
    masks_copy = masks.copy()

    for mask in masks_copy[1:]:
        edge_sobel = filters.sobel(mask)
        edge_sobel = np.where(edge_sobel > 0.5, True, False)
        mask = np.where(mask > 0.5, 1, 0)
        for i in range(3):
            image[:, :, i] = image[:, :, i] + mask * np.random.uniform(-0.8, 0.8)
            # image[:, :, i] = np.minimum(1, image[:, :, i] + mask * np.random.uniform(0.2, 0.8))
            # image[:, :, i][edge_sobel] = 0
            image[:, :, i][np.where(image[:, :, i] > 1)] = 1

    return image


def convert(y):
    seg = torch.tensor(y)
    seg = one_hot(seg.to(torch.int64), num_classes=9)
    seg = torch.permute(seg, (2, 0, 1)).numpy()

    return seg


def plot_car_segmentation(x, y, y_hat):
    seg = convert(y)

    y_hat_image = apply(x, y_hat)
    y_image = apply(x, seg)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2 * 9, 9))
    axs[0].set_title("y")
    axs[1].set_title("y_hat")

    axs[0].imshow(y_image, vmin=0, vmax=1)
    axs[1].imshow(y_hat_image, vmin=0, vmax=1)

    fig.tight_layout()
    plt.show()
