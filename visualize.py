import matplotlib.pyplot as plt
from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
import torch.nn.functional as F
from image_dataset import *
from Unet_pp_v2 import Unet_pp
import torch


def get_masks_pred(model, x):
    y_hat = model(x)
    y_hat = F.one_hot(y_hat.argmax(dim=1), 9).permute(0, 3, 1, 2).float()
    y_hat = y_hat.detach().numpy()
    print(y_hat.shape)
    return y_hat


files = files_name()
np.random.shuffle(files)

train_files = files[: int(len(files) * 0.8)]
test_files = files[int(len(files) * 0.8): int(len(files) * 0.9)]
validation_files = files[int(len(files) * 0.9):]

test_dataset = ImageDataset(test_files, size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = SegmentationModel.load_from_checkpoint(checkpoint_path="epoch=99-step=700.ckpt")

# disable randomness, dropout, etc...
model.eval()

image = np.load("carseg_data/clean_data/3.npy").astype(np.float32)
x = torch.tensor(rgb2gray(image[:3]).reshape(1, 1, 256, 256))
print(x.shape)

seg = torch.tensor(image[3])
seg = F.one_hot(seg.to(torch.int64), num_classes=9)
y = torch.permute(seg, (2, 0, 1))
y = y.detach().numpy()
print(y.shape)

y_hat = get_masks_pred(model, x)

fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(12, 54))
for i in range(9):
    for j in range(2):
        if j == 0:
            axs[i, j].imshow(y_hat[0, i], cmap='gray')
        else:
            axs[i, j].imshow(y[i], cmap='gray')
plt.show()
