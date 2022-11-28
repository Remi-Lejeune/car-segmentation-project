import random
import torchvision.transforms.functional as TF
from torchvision import transforms 
import torch
import numpy as np

def flip_im(input, output, pflip=0.2):
    # Randomly flip images horizontally

    im = torch.from_numpy(np.transpose(input, (2,0,1)))
    seg = torch.from_numpy(output)

    trans_im = TF.hflip(im)
    trans_seg = TF.hflip(seg)

    trans_im = np.transpose(trans_im.numpy(),(1,2,0))
    trans_seg = trans_seg.numpy()
    
    return trans_im, trans_seg



def zoom_im(input, output, min_crop_sz=150, pzoom=0.2):
    # Zoom in on the images randomly and rescale to the original size

    # Convert numpy arrays to tensors.
    im = torch.from_numpy(np.transpose(input, (2,0,1)))
    seg = torch.from_numpy(output)
    seg = torch.reshape(seg, (1, 256, 256))

    crop_sz = int(random.uniform(a=min_crop_sz, b=im.shape[1]))

    # Apply center crop and resize to the original size
    trans_im = transforms.CenterCrop(size=crop_sz)(im)
    trans_im = transforms.Resize(size=im.shape[1])(trans_im)
    trans_seg = transforms.CenterCrop(size=crop_sz)(seg)
    trans_seg = transforms.Resize(size=im.shape[1])(trans_seg)

    # Convert back to numpy arrays
    trans_im = np.transpose(trans_im.numpy(),(1,2,0))
    trans_seg = torch.squeeze(trans_seg)
    trans_seg = trans_seg.numpy()

    return trans_im, trans_seg




