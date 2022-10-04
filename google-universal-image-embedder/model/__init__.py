import pickle
from pathlib import Path

import pytorch_lightning as pl
import torch.nn.functional as F
from copy import deepcopy
from typing import List
from torch import (
    Tensor,
    no_grad,
    cat,
    diag,
    eye,
    logsumexp,
    mean,
    nn,
    optim,
    roll,
    unsqueeze,
)
from torchvision import models
from torchvision.transforms import functional as TF
from model.ssl_framework import BYOLLoss


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Tanh(),
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
    def __init__(self, encoder_size: int = 2, decoder_size: int = 2):

        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                EncoderStack(1024, 512, encoder_size),
                EncoderStack(512, 256, encoder_size),
                EncoderStack(256, 128, encoder_size),
                EncoderStack(128, 64, encoder_size),
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderStack(64, 128, decoder_size),
                DecoderStack(128, 256, decoder_size),
                DecoderStack(256, 512, decoder_size),
                DecoderStack(512, 1024, decoder_size),
            ]
        )
        self.head = nn.Linear(1024, 1024)

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            encoded = [x]
            decoded = []

            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encoded.append(x)

            for decoder_layer in self.decoder_layers:
                x = decoder_layer(x)
                decoded.append(x)

            decoded[-1] = self.head(x)
            return decoded[-1], encoded, decoded

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


class _Model(nn.Module):
    def __init__(self):

        super().__init__()
        self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)

        for parameter in self.vit.parameters():
            parameter.requires_grad = False

        self.vit.heads = nn.Identity()
        self.autoencoder = AutoEncoder()

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            return self.autoencoder(x)

        x = TF.resize(x, [224, 224])
        x = x / 255
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return self.autoencoder(self.vit(x))


class MLP(nn.Module):
    def __init__(self, input_size, projection_size, projection_hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class BYOLEncoder(nn.Module):
    def __init__(self, network, input_size, projection_size, projection_hidden_size):
        super().__init__()

        self.network = network
        self.projector = MLP(input_size, projection_size, projection_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(self.network(x))


class BYOLModel(nn.Module):
    def __init__(
        self,
        network,
        encode_size,
        projection_size,
        projection_hidden_size,
        target_decay_rate=0.996,
    ):
        super().__init__()

        self.network = network
        self.online_encoder = BYOLEncoder(
            network, encode_size, projection_size, projection_hidden_size
        )
        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        self.target_decay_rate = target_decay_rate

        self.target_encoder = self._get_target_encoder()

    def _get_target_encoder(self) -> MLP:
        target = deepcopy(self.online_encoder)
        for param in target.parameters():
            param.requires_grad = False
        return target

    def update_target_encoder(self):
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target_param.data = (
                target_param.data * self.target_decay_rate
                + (1 - self.target_decay_rate) * online_param.data
            )

    def forward(self, x: Tensor):
        if not self.training:
            return self.online_encoder(x)

        online_proj_1 = self.online_encoder(x[0])
        online_proj_2 = self.online_encoder(x[1])

        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with no_grad():
            target_proj_1 = self.target_encoder(x[0])
            target_proj_2 = self.target_encoder(x[1])

        return [online_pred_1, online_pred_2, target_proj_1, target_proj_2]


class BYOLLightningModule(pl.LightningModule):
    def __init__(
        self, network, encode_size, projection_size=64, projection_hidden_size=2048
    ):
        super().__init__()

        self.byol = BYOLModel(
            network, encode_size, projection_size, projection_hidden_size
        )

        self.loss = BYOLLoss()

    def forward(self, x: Tensor):
        return self.byol(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        out = self.forward(batch)
        loss = self.loss(*out)

        return loss

    def on_before_zero_grad(self, _):
        self.byol.update_target_encoder()

    def configure_optimizers(self) -> optim.Optimizer:

        return optim.Adam(self.parameters(), lr=3e-4)


class Model(pl.LightningModule):
    def __init__(self):

        super().__init__()
        self.model = _Model()

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pred, int_enc, int_dec = self.forward(batch)
        loss = sum(
            [
                F.mse_loss(enc_loss, dec_loss)
                for enc_loss, dec_loss in zip(int_enc, reversed(int_dec))
            ]
        )

        return loss

    def configure_optimizers(self) -> optim.Optimizer:

        return optim.Adam(self.parameters(), lr=0.0005)


class PCAModel(nn.Module):
    def __init__(self, pca_model: Path):

        super().__init__()
        self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()

        with pca_model.open("rb") as f:
            self.pca = wrap(pickle.load(f))

    def forward(self, x: Tensor) -> Tensor:

        x = TF.resize(x, [224, 224])
        x = x / 255
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return self.pca(self.vit(x))


class SimCLR(pl.LightningModule):
    def __init__(self, f: nn.Module, h_size: int, z_size: int, tau: float = 0.07):

        super().__init__()
        self.f = f
        self.g = nn.Sequential(
            nn.Linear(h_size, z_size),
            nn.ReLU(),
            nn.Linear(z_size, h_size),
        )
        self.tau = tau

    def compute_similarity_matrix(z: Tensor) -> Tensor:

        return F.cosine_similarity(unsqueeze(z, dim=1), unsqueeze(z, dim=0), dim=2)

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            return self.g(self.f(x[:, 0])), self.g(self.f(x[:, 1]))

        return self.f(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:

        N = batch.size(0)
        z1, z2 = self(batch)
        z = cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
        sim_mat = self.compute_similarity_matrix(z)
        pos = cat([diag(sim_mat, N), diag(sim_mat, -N)], dim=0)


if __name__ == "__main__":
    import torch

    f = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    f.fc = nn.Sequential(nn.Linear(2048, 64), nn.Tanh())
    simclr = SimCLR(f, 64, 256)

    print(simclr.training_step(torch.randn(16, 2, 3, 224, 224), 0))
