from typing import Tuple

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, nn, no_grad, optim
from torchvision import models


class EncoderBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, n: int):

        super().__init__()
        self.linears = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_features, in_features), nn.GELU())
                for _ in range(n - 1)
            ]
        )
        self.last = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:

        return self.last(self.linears(x))


class DecoderBlock:
    def __init__(self, in_features: int, out_features: int, n: int):

        super().__init__()
        self.first = nn.Linear(in_features, out_features)
        self.linears = nn.Sequential(
            *[
                nn.Sequential(nn.GELU(), nn.Linear(out_features, out_features))
                for _ in range(n - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.linears(self.first(x))


class AutoEncoder(nn.Module):
    def __init__(self, encoder_size: int = 2, decoder_size: int = 2):

        super().__init__()
        self.encoder = nn.Sequential(
            EncoderBlock(1024, 512, encoder_size),
            nn.GELU(),
            EncoderBlock(512, 256, encoder_size),
            nn.GELU(),
            EncoderBlock(256, 128, encoder_size),
            nn.GELU(),
            EncoderBlock(128, 64, encoder_size),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            DecoderBlock(64, 128, decoder_size),
            nn.GELU(),
            DecoderBlock(128, 256, decoder_size),
            nn.GELU(),
            DecoderBlock(256, 512, decoder_size),
            nn.GELU(),
            DecoderBlock(512, 1024, decoder_size),
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
