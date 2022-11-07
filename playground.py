import numpy as np
from matplotlib import pyplot as plt
from skimage import io

data = np.load('carseg_data/clean_data/1.npy')
data_a = np.load('carseg_data/clean_data/0_a.npy')
print(data.shape)
plt.figure()
io.imshow(data_a[:3])
plt.show()