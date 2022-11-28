from image_dataset import *
import matplotlib.pyplot as plt
from augmentation import *


def plot_transformed_img(input, output):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 16))
    axs[0].imshow(input)
    axs[0].set_title("Image")
    axs[1].imshow(output)
    axs[1].set_title("Mask")
    plt.show()


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, rgb.shape[1], rgb.shape[2])
files = files_name()


angle = np.random.randint(0, 360)
image = np.load("carseg_data/clean_data/31.npy").astype(np.float32)

input, output = zoom_im(image[:3], image[3])
plot_transformed_img(np.transpose(input, (1, 2, 0)), output)
plot_transformed_img(np.transpose(flip_im(input, output)[0], (1, 2, 0)), flip_im(input, output)[1])






