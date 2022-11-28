import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import one_hot
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, Softmax
from torch.nn.functional import cross_entropy
from unet_model import *



class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.network = UNet(n_channels=1, n_classes=9)
        self.loss = DiceLoss()

    def training_step(self, batch, batch_nb):
        x, y = batch
        x_hat = self.network(x).float()
        loss = self.loss(F.softmax(x_hat, dim=1).float(), y.float()) + F.cross_entropy(x_hat.float(), y.float())
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.network(x).float()
        loss = self.loss(F.softmax(x_hat, dim=1).float(), y.float()) + F.cross_entropy(x_hat.float(), y.float())
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.network(x).float()
        loss = self.loss(F.softmax(x_hat, dim=1).float(), y.float()) + F.cross_entropy(x_hat.float(), y.float())
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x):
        return self.network.forward(x)



class DiceLoss(nn.Module):
    """
    This module implements the Dice loss function used to train the model
    """
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Method used to compute the Dice Loss function
        :param input: the input is the probabilities for each mask of shape (n_classes, height, width)
        :param target_one_hot: output masks of shape (n_classes, height, width)
        :return: the computed value of the Dice Loss
        """
        dims = (1, 2, 3)
        intersection = torch.sum(input * target_one_hot, dims)
        cardinality = torch.sum(input + target_one_hot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)
