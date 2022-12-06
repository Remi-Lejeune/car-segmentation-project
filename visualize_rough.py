import matplotlib.pyplot as plt
from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
import torch.nn.functional as F
from image_dataset import *
import torch
from dice_loss import DiceLoss


def get_masks_pred(model, x):
    y_hat = model(x)
    y_hat = F.softmax(y_hat, dim=1)
    y_hat = F.one_hot(y_hat.argmax(dim=1), 9).permute(0, 3, 1, 2).float()
    return y_hat


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
    plt.savefig(title+".svg")


def get_masks_out(y):
    seg = torch.tensor(y)
    seg = F.one_hot(seg.to(torch.int64), num_classes=9)
    masks = torch.permute(seg, (2, 0, 1))
    masks = masks.detach().numpy()
    return masks


test_files = get_test_files()
test_dataset = ImageDataset(test_files, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = SegmentationModel.load_from_checkpoint(checkpoint_path="epoch=299-step=20400.ckpt")
# disable randomness, dropout, etc...
model.eval()

dice_losses = []
loss = DiceLoss()
it = iter(test_dataloader)

for x, y in it:
    y_hat = get_masks_pred(model, x)
    dice_loss = loss(y_hat, y)
    dice_losses.append(dice_loss)

worst_pred_idx = np.argmax(dice_losses)
print(np.max(dice_losses))
print(np.min(dice_losses))
best_pred_idx = np.argmin(dice_losses)
worst_filepath = test_files[worst_pred_idx]
best_filepath = test_files[best_pred_idx]

image = np.load(worst_filepath).astype(np.float32)
y = get_output_masks(image[3])
y_hat = get_masks_pred(model, torch.tensor(image[:3].reshape(1, 3, 256, 256)))
y_hat = y_hat.detach().numpy()

y_hat_categorical = np.argmax(y_hat, axis=1)

np.save("worst_sample", y_hat_categorical)

plot_output_and_pred(y_hat, y, title="Worst sample")

image = np.load(best_filepath).astype(np.float32)
y = get_output_masks(image[3])
y_hat = get_masks_pred(model, torch.tensor(image[:3].reshape(1, 3, 256, 256)))
y_hat = y_hat.detach().numpy()

y_hat_categorical = np.argmax(y_hat, axis=1)
np.save("best_sample", y_hat_categorical)
plot_output_and_pred(y_hat, y, title="Best sample")

file_dict = {"best": best_filepath, "worst": worst_filepath}

with open("files", "w") as f:
    f.write(file_dict.__str__())



