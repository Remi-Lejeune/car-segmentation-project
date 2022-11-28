import numpy as np
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss

from unet_model import *
from dice_loss import DiceLoss


class SegmentationModel(pl.LightningModule):
    def __init__(self, weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.network = UNet(n_channels=3, n_classes=9)
        self.dice_loss = DiceLoss(weights=weights)
        self.loss = CrossEntropyLoss(weight=weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            N = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            module.weight.data.normal_(mean=0.0, std=np.sqrt((2 / N)))

        if isinstance(module, nn.ConvTranspose2d):
            N = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            module.weight.data.normal_(mean=0.0, std=np.sqrt((2 / N)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.network(x).float()
        loss = self.dice_loss(F.softmax(y_hat, dim=1), y.float()) + self.cross_entropy(y_hat.float(), y.float())
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x).float()

        # Use dice score here
        loss = 1.0 - self.dice_loss(F.softmax(y_hat, dim=1).float(), y.float())

        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x).float()
        loss = self.dice_loss(F.softmax(y_hat, dim=1).float(), y.float()) + CrossEntropyLoss(y_hat.float(), y.float())

        self.log("validation test score", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.network.forward(x)
