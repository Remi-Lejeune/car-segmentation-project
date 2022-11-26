from image_dataset import *
from unet_model import UNet



def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, rgb.shape[1], rgb.shape[2])
files = files_name()



image = np.load("carseg_data/clean_data/0_a.npy").astype(np.float32)
x = torch.tensor(image[:3].reshape(1, 3, 256, 256))
model = UNet(n_channels=3, n_classes=9)
y_hat = model(x)
print(y_hat.shape)





#images_files = photos_file_name()
#files = get_npy_filenames(photos_file_name())



#plt.imshow(rgb2gray(image), cmap='gray')
#plt.show()



