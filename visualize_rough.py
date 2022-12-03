import matplotlib.pyplot as plt
from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
import torch.nn.functional as F
from image_dataset import *
import torch


def get_masks_pred(model, x):
    y_hat = model(x)
    y_hat = F.one_hot(y_hat.argmax(dim=1), 9).permute(0, 3, 1, 2).float()
    y_hat = y_hat.detach().numpy()
    return y_hat.astype(int)


def plot_output_and_pred(y_hat, y, title):
    fig, axs = plt.subplots(nrows=9, ncols=3, figsize=(12, 54))
    for i in range(9):
        axs[i, 0].imshow(y_hat[0, i], cmap='gray')
        axs[i, 0].set_title(f"Y_hat {i}")
        axs[i, 1].imshow(y[i], cmap='gray')
        axs[i, 1].set_title(f"Y {i}")
        pos_neg = axs[i, 2].imshow(y_hat[0, i] - y[i], vmin=-1, vmax=1, cmap='RdBu', interpolation=None)
        axs[i, 2].set_title(f"Y_hat_{i} - Y_{i}")
        cbar = fig.colorbar(pos_neg, ax=axs[i, 2], extend=None)
    plt.suptitle(title.split("/")[-1].split(".")[0])
    fig.tight_layout()
    fig.subplots_adjust(top=0.99)
    plt.show()


def get_masks_out(y):
    seg = torch.tensor(y)
    seg = F.one_hot(seg.to(torch.int64), num_classes=9)
    masks = torch.permute(seg, (2, 0, 1))
    masks = masks.detach().numpy()
    return masks


test_files = get_test_files()
files = get_clean_files()
#print(files)

path = "carseg_data/clean_data/0_a.npy"

print(path in files)


"""
model = SegmentationModel.load_from_checkpoint(checkpoint_path="epoch=999-step=70000.ckpt")

# disable randomness, dropout, etc...
model.eval()

dice_losses = []

for file in test_files:
    image = np.load(file).astype(np.float32)
    x = torch.tensor(image[:3].reshape(1, 3, 256, 256))
    y = get_masks_out(image[3])
    y_hat = get_masks_pred(model, x)

    eps = 10e-6
    dims = (1, 2, 3)
    intersection = np.sum(y_hat * y, dims)
    cardinality = np.sum(y_hat + y, dims)
    dice_score = 2. * intersection / (cardinality + eps)
    dice_losses.append(np.mean(1. - dice_score))

    # plot_output_and_pred(y_hat, y, title=file)


worst_pred_idx = np.argmax(dice_losses)
print(test_files[worst_pred_idx])


image = np.load(test_files[worst_pred_idx]).astype(np.float32)
x = torch.tensor(image[:3].reshape(1, 3, 256, 256))
y = get_masks_out(image[3])
y_hat = get_masks_pred(model, x)
plot_output_and_pred(y_hat, y, title=test_files[worst_pred_idx])
"""