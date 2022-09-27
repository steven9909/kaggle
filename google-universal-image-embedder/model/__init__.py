import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor, nn, optim, mean, FloatTensor
from torchvision import models
from torchvision.transforms import functional as TF


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
