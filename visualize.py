import matplotlib.pyplot as plt

from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from torch.nn.functional import one_hot
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

y = y.detach().numpy()
#y_hat = torch.argmax(y_hat, dim=1)
#y_hat = one_hot()
y_hat = y_hat.detach().numpy()

plt.imshow(y_hat[0, 0], vmin=0, vmax=1, cmap='gray')
plt.show()

"""
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
axs = axs.flatten()
for i in range(9):
    axs[i].imshow(y_hat[0, i])
plt.show()
"""