import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import numpy as np

from Unet_pp_v2 import Unet_pp


class SegmentationModel(pl.LightningModule):
    def __init__(self, weights):
        super().__init__()
        self.save_hyperparameters()
        self.network = Unet_pp(256, 256, 3, 9)
        self.loss = nn.CrossEntropyLoss(weight=weights)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            N=module.in_channels*module.kernel_size[0]*module.kernel_size[1]
            module.weight.data.normal_(mean=0.0, std=np.sqrt((2/N)))

        if isinstance(module, nn.ConvTranspose2d):
            N=module.in_channels*module.kernel_size[0]*module.kernel_size[1]
            module.weight.data.normal_(mean=0.0, std=np.sqrt((2/N)))

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


