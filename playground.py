import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from image_dataset import files_name, ImageDataset

files = files_name()
train_dataset = ImageDataset(files, size=200)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# for batch_ndx, sample in enumerate(train_dataloader):
#     print(sample[0].shape)

image = np.load('carseg_data/clean_data/DOOR_0003.npy')
print(image[3][image[3] == 4])
plt.show()
plt.imshow(image[3])

plt.show()