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
    print(y_hat.shape)
    return y_hat.astype(int)


files = files_name()
np.random.shuffle(files)

train_files = files[: int(len(files) * 0.8)]
test_files = files[int(len(files) * 0.8): int(len(files) * 0.9)]
validation_files = files[int(len(files) * 0.9):]

test_dataset = ImageDataset(test_files, size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = SegmentationModel.load_from_checkpoint(checkpoint_path="epoch=999-step=31000.ckpt")

# disable randomness, dropout, etc...
model.eval()

image = np.load("carseg_data/clean_data/21_a.npy").astype(np.float32)
x = torch.tensor(rgb2gray(image[:3]).reshape(1, 1, 256, 256))
print(x.shape)

seg = torch.tensor(image[3])
seg = F.one_hot(seg.to(torch.int64), num_classes=9)
y = torch.permute(seg, (2, 0, 1))
y = y.detach().numpy()
print(y.shape)

y_hat = get_masks_pred(model, x)

from matplotlib.colors import ListedColormap
import seaborn as sns

fig, axs = plt.subplots(nrows=9, ncols=3, figsize=(12, 54))
for i in range(9):
    axs[i, 0].imshow(y_hat[0, i], cmap='gray')
    axs[i, 0].set_title(f"Y_hat {i}")
    axs[i, 1].imshow(y[i], cmap='gray')
    axs[i, 1].set_title(f"Y {i}")
    pos_neg = axs[i, 2].imshow(y_hat[0, i]-y[i], vmin=-1, vmax=1, cmap='RdBu', interpolation=None)
    axs[i, 2].set_title(f"Y_hat_{i} - Y_{i}")
    cbar = fig.colorbar(pos_neg, ax=axs[i, 2], extend=None)
plt.tight_layout()
plt.show()