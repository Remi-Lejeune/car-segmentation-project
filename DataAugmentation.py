import random
import torchvision.transforms.functional as TF
from torchvision import transforms 
import torch
import numpy as np

def flip_im(input, output):
    # Randomly flip images horizontally
    im = torch.from_numpy(input)
    seg = torch.from_numpy(output)
    trans_im = TF.hflip(im)
    trans_seg = TF.hflip(seg)
    trans_im = trans_im.numpy()
    trans_seg = trans_seg.numpy()
    return trans_im, trans_seg


def zoom_im(input, output, min_crop_sz=150):
    # Zoom in on the images randomly and rescale to the original size
    n_channels = input.shape[0]
    H, W = input.shape[1], input.shape[2]
    # Convert numpy arrays to tensors.
    im = torch.from_numpy(input).reshape(n_channels, H, W)
    seg = torch.from_numpy(output).reshape(1, H, W)
    seg = torch.reshape(seg, (1, H, W))

    crop_sz = int(random.uniform(a=min_crop_sz, b=im.shape[1]))

    # Apply center crop and resize to the original size
    trans_im = transforms.CenterCrop(size=crop_sz)(im)
    trans_im = transforms.Resize(size=im.shape[1])(trans_im)
    trans_seg = transforms.CenterCrop(size=crop_sz)(seg)
    trans_seg = transforms.Resize(size=im.shape[1])(trans_seg)

    # Convert back to numpy arrays
    trans_im = trans_im.numpy()
    trans_seg = torch.squeeze(trans_seg)
    trans_seg = trans_seg.numpy()

    return trans_im, trans_seg




