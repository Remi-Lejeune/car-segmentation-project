import torch
from unet_model import *
from image_dataset import *
import matplotlib.pyplot as plt
from augmentation import *
from dice_loss import *


def plot_transformed_img(input, output):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 16))
    axs[0].imshow(input)
    axs[0].set_title("Image")
    axs[1].imshow(output)
    axs[1].set_title("Mask")
    plt.show()


def plot_masks(masks):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(27, 27))
    axs = axs.flatten()
    for i in range(9):
        axs[i].imshow(masks[i], cmap='gray')
        axs[i].set_title(f"Mask n.{i}")
    plt.show()


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, rgb.shape[1], rgb.shape[2])



files = get_clean_files()
np.random.seed(0)
np.random.shuffle(files)
print(files[int(len(files)*0.9):])



"""
image = np.load("carseg_data/clean_data/31.npy").astype(np.float32)
gray_scaled_img = rgb2gray(image[:3])

# Apply the augmentation function
input, output = flip_im(gray_scaled_img, image[3])

# Check sizes
print(f"Input shape = {input.shape}, Output shape = {output.shape}")

# Show the results
plot_transformed_img(np.transpose(input, (1, 2, 0)), output)
masks = get_output_masks(output)
print(f"Masks shape = {masks.shape}")
plot_masks(masks)


input = torch.tensor(input).reshape(1, 1, 256, 256)
masks = torch.tensor(masks)
print(f"Masks shape{masks.shape}")
net = UNet(n_channels=1, n_classes=9)
y_hat = net(input)
print(f"Y_hat shape = {y_hat.shape}")
loss = DiceLoss()
print(loss(y_hat, masks))
"""







