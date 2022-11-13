import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms

from image_dataset import ImageDataset, files_name
from segmentation_model import SegmentationModel
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init


from torch.nn.parameter import Parameter
from torchvision.datasets import MNIST

mnist_trainset = MNIST("./temp/", train=True, download=True, transform=ToTensor())
mnist_testset = MNIST("./temp/", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(mnist_trainset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_testset, batch_size=32, shuffle=True)
x_train = mnist_trainset.data[:1000].view(-1, 784).float()

#Hyperparameters
num_l1 = 512
num_features = 25088

num_classes = 10
dims = (1, 28, 28)
channels, width, height = dims
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
hidden_size = 64
# define network
model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

model = SegmentationModel(model)

trainer = Trainer(
    accelerator="auto",
    max_epochs=3,
    min_epochs=1,
)

trainer.fit(model, train_dataloader)
trainer.test(model, test_dataloader)
