import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import one_hot
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, Softmax
from torch.nn.functional import cross_entropy

from Unet_pp_v2 import Unet_pp


class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = Unet_pp(256, 256, im_channels=1, num_features=9)
        self.loss = DiceLoss()

    def training_step(self, batch, batch_nb):
        x, y = batch
        x_hat = self.network(x).float()
        loss = self.loss((x_hat), y.float())

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.network(x).float()
        loss = self.loss((x_hat), y.float())

        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.network(x).float()
        print("x_hat", x_hat.shape)
        print("y", y.shape)
        loss = self.loss((x_hat), y.float())

        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        return self.network.forward(x)


class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        # compute the actual dice score
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        # target_one_hot = one_hot(target, num_classes=input.shape[1])
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)
