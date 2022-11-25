import matplotlib.pyplot as plt

from segmentation_model import SegmentationModel
import numpy as np
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from Unet_pp_v2 import Unet_pp
import torch

files = [files_name()[0]]

test_dataset = ImageDataset(files, size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = SegmentationModel.load_from_checkpoint(checkpoint_path="epoch=759-step=760.ckpt")
#
# # disable randomness, dropout, etc...
model.eval()

x, y = next(iter(test_dataloader))

# predict with the model
y_hat = model(x)

y_hat = y_hat.detach().numpy()
y = y.detach().numpy()

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
axs = axs.flatten()
print(y.shape)
for i in range(9):
    axs[i].imshow(y_hat[0, i], vmin=0, vmax=1)
    print(y_hat[0, i])
plt.show()
# #
# plt.figure()
# plt.imshow((x[0][0]))
# plt.show()
#
plt.figure()
# plt.imshow((y[0]))
# plt.show()
# print(y[0][y[0] > 1])
# print(y.shape)
# print(x.shape)
