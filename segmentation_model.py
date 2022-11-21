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
        return self.helper(batch, batch_nb, "train_loss")

    def test_step(self, batch, batch_idx):
        return self.helper(batch, batch_idx, "test_loss")

    def validation_step(self, batch, batch_idx):
        return self.helper(batch, batch_idx, "val_loss")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def forward(self, x):
        return self.network.forward(x)

    def helper(self, batch, batch_idx, mode):
        x, y = batch
        x_hat = self.network(x)
        loss = self.loss(x_hat[:, 1:], y.long())

        self.log(mode, loss)
        return loss
