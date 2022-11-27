from image_dataset import *
import skimage.transform as transform
import matplotlib.pyplot as plt
from augmentation import *

from unet_model import UNet


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, rgb.shape[1], rgb.shape[2])
files = files_name()


angle = np.random.randint(0, 360)
image = torch.from_numpy(np.load("carseg_data/clean_data/31.npy").astype(np.float32))
gray_image = rgb2gray(image[:3])
mask = image[3]
transf_matrix = transform.SimilarityTransform(scale=2)
input = transform.warp(gray_image, transf_matrix)
output = transform.warp(mask, transf_matrix)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 16))
axs[0].imshow(input.reshape(256, 256))
axs[0].set_title("Image")
axs[1].imshow(output)
axs[1].set_title("Mask")
plt.show()



"""
transformed_input = torch.from_numpy(transform.rotate(gray_image.reshape(256, 256), angle))
transformed_output = transform.rotate(image[3].reshape(256, 256), angle)

output = torch.tensor(transformed_output)
masks = F.one_hot(output.to(torch.int64), num_classes=9)
y = torch.permute(masks, (2, 0, 1))
y = y.detach().numpy()


fig, axs = plt.subplots(nrows=3, ncols=3)
axs = axs.flatten()


for i in range(y.shape[0]):
    axs[i].imshow(y[i])
    axs[i].set_title(f"Mask {i+1}")
plt.tight_layout()
plt.show()

print(angle)
net = UNet(n_channels=1, n_classes=9)
out = net(transformed_input.reshape(1, 1, 256, 256))


x = torch.tensor(image[:3].reshape(1, 3, 256, 256))
model = UNet(n_channels=3, n_classes=9)
y_hat = model(x)
print(y_hat.shape)
images_files = photos_file_name()
files = get_npy_filenames(photos_file_name())
plt.imshow(rgb2gray(image), cmap='gray')
plt.show()
"""


