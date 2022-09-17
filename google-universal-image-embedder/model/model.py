from torchvision import models
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from torch import no_grad


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.vit_l_16(weights=models.ViT_L_16_Weights)
        self.model.heads = nn.Identity()

        self.autoencoder = AutoEncoder()

    def forward(self, x):
        with no_grad():
            y = self.model(x)
        y_hat = self.autoencoder(y)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self(batch)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
