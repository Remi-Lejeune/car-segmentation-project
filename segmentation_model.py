import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from unet_test import UNet

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.network = UNet(3, 9)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_nb):
        return self.helper(batch, batch_nb, "train_loss")

    def test_step(self, batch, batch_idx):
        self.helper(batch, batch_idx, "test_loss")

    def validation_step(self, batch, batch_idx):
        self.helper(batch, batch_idx, "val_loss")

    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=0.02)

    def forward(self, x):
        return self.network.forward(x)

    def helper(self, batch, batch_idx, mode):
        x, y = batch
        x_hat = self.network(x)
        # x_hat = x_hat.permute(0, 2, 3, 1).contiguous().view(-1, x_hat.size(1))
        # y = y.permute(1, 2, 0).contiguous().view(-1).long()
        # print(x_hat.shape)
        # print(y.shape)

        loss = self.loss(x_hat.float(), y.long())
        self.log(mode, loss)
        return loss
