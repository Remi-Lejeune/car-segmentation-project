import skimage.transform as transform
import numpy.random as random
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms

def rotate_image(input, output, H=256, W=256, angle=None):
    """
    Applies rotation to the image randomly choosing the angle.
    It requires gray-scaled images as input
    :param input: the input gray-scaled image
    :param output: mask containing all the features
    :return: the rotated input and output with shape (H, W)
    """
    if angle is None:
        angle = random.randint(-90, 90)
    transf_input = transform.rotate(input.reshape(H, W), angle)
    transf_output = transform.rotate(output.reshape(H, W), angle)
    return transf_input, transf_output


def similarity_transform_image(input, output, H=256, W=256, point=None):
    """
    Performs similarity transformation on a gray-scaled image
    :param input: gray-scale image
    :param output: mask containing all the features
    :param H: height of the image
    :param W: width of the image
    :return: the transformed input and output with shape (H, W)
    """
    if point is None:
        ax_0 = np.random.randint(-H/4, H/4)
        ax_1 = np.random.randint(-H/4, W/4)
    else:
        ax_0 = point[0]
        ax_1 = point[1]
    transf_matrix = transform.SimilarityTransform(translation=(ax_0, ax_1))
    transf_input = transform.warp(input.reshape(H, W), transf_matrix)
    transf_output = transform.warp(output.reshape(H, W), transf_matrix)
    return transf_input, transf_output


def flip_im(input, output):
    """
    Flip the input image and the corresponding mask
    :param input: image with shape (n_channels, H, W)
    :param output: image with shape (H, W)
    :return: the transformed input and output
    """
    # Randomly flip images horizontally
    im = torch.from_numpy(input)
    seg = torch.from_numpy(output)
    trans_im = TF.hflip(im)
    trans_seg = TF.hflip(seg)
    trans_im = trans_im.numpy()
    trans_seg = trans_seg.numpy()
    return trans_im, trans_seg


def zoom_im(input, output, min_crop_sz=150):
    """
    Zoom in the input image and the corresponding mask
    :param input: image with shape (n_channels, H, W)
    :param output: image with shape (H, W)
    :return: the transformed input and output
    """
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