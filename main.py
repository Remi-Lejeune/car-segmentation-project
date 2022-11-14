import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from segmentation_model import SegmentationModel
from Unet_pp_v2 import Unet_pp
# from torchsummary import summary

files = files_name()
np.random.shuffle(files)

train_files = files[: int(len(files) * 0.8)]
test_files = files[int(len(files) * 0.8): int(len(files) * 0.9)]
validation_files = files[int(len(files) * 0.9):]

train_dataset = ImageDataset(train_files, size=200)
test_dataset = ImageDataset(test_files, size=200)
validation_dataset = ImageDataset(validation_files, size=200)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

model = SegmentationModel(Unet_pp(256, 256, 1, 1))
# summary(Unet_pp(256, 256, 1, 1), (1, 256, 256))

trainer = Trainer(
    max_epochs=10,
    min_epochs=5,
    overfit_batches=1
)

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)
