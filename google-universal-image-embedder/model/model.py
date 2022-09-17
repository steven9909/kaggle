from typing import Tuple

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, nn, no_grad, optim
from torchvision import models


class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_last: bool = False):

        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)

        self.is_last = is_last

    def forward(self, x):

        x = F.gelu(self.linear1(x))
        return self.linear2(x) if self.is_last else F.gelu(self.linear2(x))


class AutoEncoder(nn.Module):
    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(
            LinearBlock(1024, 512),
            LinearBlock(512, 256),
            LinearBlock(256, 128),
            LinearBlock(128, 64),
        )
        self.decoder = nn.Sequential(
            LinearBlock(64, 128),
            LinearBlock(128, 256),
            LinearBlock(256, 512),
            LinearBlock(512, 1024, is_last=True),
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.decoder(self.encoder(x))


class Model(pl.LightningModule):
    def __init__(self):

        super().__init__()
        self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights)
        self.vit.heads = nn.Identity()
        self.autoencoder = AutoEncoder()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        with no_grad():
            y = self.vit(x)

        y_hat = self.autoencoder(y)

        return y, y_hat

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = F.mse_loss(*self.forward(batch))
        self.log("loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:

        return optim.Adam(self.parameters(), lr=0.001)
