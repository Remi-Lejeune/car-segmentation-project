import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from image_dataset import files_name, ImageDataset

files = files_name()
train_dataset = ImageDataset(files, size=200)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch_ndx, sample in enumerate(train_dataloader):
    print(sample[0].shape)
