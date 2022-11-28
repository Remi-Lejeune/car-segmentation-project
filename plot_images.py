import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from segmentation_model import SegmentationModel
from Unet_pp_v2 import Unet_pp
from DataAugmentation import DataAugmentation
import sklearn
#from Unet_v2 import *
from unet_model import UNet

import matplotlib.pyplot as plt

# Load image
im_seg = np.load("../../../../Desktop/clean_data/69.npy")
im = np.transpose(im_seg[:3], (1,2,0))
seg = im_seg[3]

data = [[im,seg]]

plt.imshow(im, vmin=0, vmax=1)
plt.show()


# Load image
im_seg2 = np.load("../../../../Desktop/clean_data/401.npy")
im2 = np.transpose(im_seg2[:3], (1,2,0))
seg2 = im_seg2[3]
plt.imshow(im2, vmin=0, vmax=1)
plt.show()


data.append([im2, seg2])

print(data)

data = DataAugmentation(data, pflip=0.2, pcrop=0.2, min_crop_sz=200, prot=0.2, max_rot_ang=30)

for i in range(len(data)):
    plt.imshow(data[i][0])
    plt.show()

