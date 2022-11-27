import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from image_dataset import *
from segmentation_model import SegmentationModel
from Unet_pp_v2 import Unet_pp
from DataAugmentation import DataAugmentation


#files = files_name()
files = get_npy_filenames(photos_file_name())
np.random.shuffle(files)

train_files = files[:int(len(files) * 0.5)]
test_files = files[int(len(files) * 0.5): int(len(files) * 0.75)]
validation_files = files[int(len(files) * 0.75):]

train_dataset = ImageDataset(train_files)
test_dataset = ImageDataset(test_files)
validation_dataset = ImageDataset(validation_files)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)


model = SegmentationModel()

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=5,
    min_epochs=1,
)

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)
