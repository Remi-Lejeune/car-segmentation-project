import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import cross_entropy

from Unet_pp_v2 import Unet_pp


class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = Unet_pp(256, 256, 3, 9)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_nb):
        x, y = batch
        x_hat = self.network(x).flatten(start_dim=2).float()
        loss = self.loss((x_hat), y.flatten(start_dim=2).float())

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.network(x).flatten(start_dim=2).float()
        loss = self.loss((x_hat), y.flatten(start_dim=2).float())

        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.network(x).flatten(start_dim=2).float()
        print("x_hat", x_hat.shape)
        print("y", y.shape)
        loss = self.loss((x_hat), y.flatten(start_dim=2).float())

        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def forward(self, x):
        return self.network.forward(x)


