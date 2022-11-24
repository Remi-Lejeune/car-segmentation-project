import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from image_dataset import ImageDataset, files_name
from segmentation_model import SegmentationModel
from Unet_pp_v2 import Unet_pp
from DataAugmentation import DataAugmentation
import sklearn

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



# Compute class weight from the trianing data.
it = iter(train_dataloader)
weights=np.zeros(9)
for  batch_im, batch_label in it: 

    for i in range(batch_label.size()[0]):
        label = batch_label[i,:,:,:]

        total_pixels = torch.sum(label, dim=(1,2)).numpy()
        total_pixels[total_pixels == 0] += 1
        weights += label.size()[1] / (9*total_pixels)
                
weights= weights/len(train_dataset.files)
weights = torch.from_numpy(weights)



model = SegmentationModel(weights=weights)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=10,
    min_epochs=5,
)

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)
