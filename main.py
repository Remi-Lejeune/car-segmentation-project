import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from segmentation_model import SegmentationModel
from Unet_pp_v2 import Unet_pp
from DataAugmentation import DataAugmentation


files = files_name()
np.random.shuffle(files)

train_files = files[: int(len(files) * 0.8)]
test_files = files[int(len(files) * 0.8): int(len(files) * 0.9)]
validation_files = files[int(len(files) * 0.9):]

train_dataset = ImageDataset(train_files)
test_dataset = ImageDataset(test_files)
validation_dataset = ImageDataset(validation_files)

# Augment the training and validation data
# train_dataset = DataAugmentation(train_dataset)
# validation_dataset = DataAugmentation(validation_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=True)


model = SegmentationModel(Unet_pp(256, 256, 3, 9))

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=10,
    min_epochs=5,
)

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)
