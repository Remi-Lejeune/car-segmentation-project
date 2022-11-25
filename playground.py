from image_dataset import *
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(1, rgb.shape[1], rgb.shape[2])



#images_files = photos_file_name()
#files = get_npy_filenames(photos_file_name())
files = files_name()


image = np.load(files[0])
print(rgb2gray(image).shape)
#plt.imshow(rgb2gray(image), cmap='gray')
#plt.show()



