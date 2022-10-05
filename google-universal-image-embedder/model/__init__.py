import pickle
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Union
from pyparsing import Optional

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, jit, nn, no_grad, optim
from torchvision import models
from torchvision.transforms import functional as TF

from model.contrastive_losses import BYOLLoss


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
            nn.Linear(input_size, projection_hidden_size, bias=False),
            nn.BatchNorm1d(projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, projection_hidden_size, bias=False),
            nn.BatchNorm1d(projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, projection_size, bias=False),
            nn.BatchNorm1d(projection_size, affine=False),
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

    @jit.unused
    def forward_training(
        self, x1: Tensor, x2: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        online_proj_1 = self.online_encoder(x1)
        online_proj_2 = self.online_encoder(x2)

        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with no_grad():
            target_proj_1 = self.target_encoder(x1)
            target_proj_2 = self.target_encoder(x2)

        return online_pred_1, online_pred_2, target_proj_2, target_proj_1

    def forward(self, x: Tensor) -> Tensor:

        x = TF.resize(x, [224, 224])
        x = x / 255
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return self.online_encoder(x)


class BYOLLightningModule(pl.LightningModule):
    def __init__(
        self, network, encode_size, projection_size=64, projection_hidden_size=2048
    ):
        super().__init__()

        self.model = BYOLModel(
            network, encode_size, projection_size, projection_hidden_size
        )

        self.loss = BYOLLoss()

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:

        return self.loss(*self.model.forward_training(*batch))

    def on_before_zero_grad(self, _):
        self.model.update_target_encoder()

    def configure_optimizers(self) -> optim.Optimizer:

        return optim.Adam(self.parameters(), lr=5e-4)


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
