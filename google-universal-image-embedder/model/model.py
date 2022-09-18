from typing import Tuple

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, nn, no_grad, optim
from torchvision import models


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)


class EncoderStack(nn.Module):
    def __init__(self, in_features: int, out_features: int, n: int):

        assert n > 0, f"n should be greater than zero, but got {n}"

        super().__init__()
        linears = []

        for _ in range(n - 1):
            linears.append(Linear(in_features, in_features))

        linears.append(Linear(in_features, out_features))

        self.model = nn.Sequential(*linears)

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)


class DecoderStack(nn.Module):
    def __init__(self, in_features: int, out_features: int, n: int):

        assert n > 0, f"n should be greater than zero, but got {n}"

        super().__init__()
        linears = []

        linears.append(Linear(in_features, out_features))

        for _ in range(n - 1):
            linears.append(Linear(out_features, out_features))

        self.model = nn.Sequential(*linears)

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_size: int = 1, decoder_size: int = 1):

        super().__init__()
        self.encoder = nn.Sequential(
            EncoderStack(1024, 512, encoder_size),
            EncoderStack(512, 256, encoder_size),
            EncoderStack(256, 128, encoder_size),
            EncoderStack(128, 64, encoder_size),
        )
        self.decoder = nn.Sequential(
            DecoderStack(64, 128, decoder_size),
            DecoderStack(128, 256, decoder_size),
            DecoderStack(256, 512, decoder_size),
            DecoderStack(512, 1024, decoder_size),
        )
        self.head = nn.Linear(1024, 1024)

    def forward(self, x: Tensor) -> Tensor:

        return self.head(self.decoder(self.encoder(x)))


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

        y, y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y)

        return loss

    def configure_optimizers(self) -> optim.Optimizer:

        return optim.Adam(self.parameters(), lr=0.0005)


if __name__ == "__main__":
    print(AutoEncoder(2, 2))
