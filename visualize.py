import matplotlib.pyplot as plt

from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from Unet_pp_v2 import Unet_pp
import torch


files = files_name()
np.random.shuffle(files)

train_files = files[: int(len(files) * 0.8)]
test_files = files[int(len(files) * 0.8): int(len(files) * 0.9)]
validation_files = files[int(len(files) * 0.9):]

test_dataset = ImageDataset(test_files, size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = SegmentationModel.load_from_checkpoint(checkpoint_path="epoch=9-step=1760.ckpt")

# disable randomness, dropout, etc...
model.eval()

x, y = next(iter(test_dataloader))

# predict with the model
y_hat = model(x)

y_hat = y_hat.detach().numpy()
y = y.detach().numpy()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
axs = axs.flatten()
for i in range(9):
    axs[i].imshow(y_hat[0, i])
plt.show()