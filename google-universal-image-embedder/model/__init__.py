from typing import List

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from torch import BoolTensor, Tensor, cat, optim

from model.shift import rshift_2d

from model.transformer.model import ViT


def construct_mask(seq_len: int) -> BoolTensor:

    return BoolTensor(np.tril(np.ones((1, seq_len, seq_len))))


class Model(pl.LightningModule):
    """
    Model

    Args:
        image_size (int): Image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        n_heads (int): Number of attention heads
        d_token (int): Dimension of each token
        n_enc_layers (int): Number of encoder layers
        n_dec_layers (int): Number of decoder layers
        seq_len (int): Sequence length
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        n_heads: int,
        d_token: int,
        n_enc_layers: int,
        n_dec_layers: int,
        seq_len: int,
    ):

        super().__init__()

        self.patch_size = patch_size

        mask = construct_mask(seq_len)
        self.register_buffer("mask", mask)
        self.model = ViT(
            d_token=d_token,
            heads=n_heads,
            n_encoder=n_enc_layers,
            n_decoder=n_dec_layers,
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=in_channels,
            image_size=image_size,
        )
        self.loss = nn.MSELoss()

    def forward(self, x: Tensor, mask: BoolTensor) -> Tensor:
        """
        Forward pass of Model

        Note:
            N: batch size
            S: sequence length
            C: channel size
            H: image height
            W: image width

        Args:
            x (Tensor): Input tensor of shape (N, C, S * H, W)
            mask (BoolTensor): Input mask

        Returns:
            Tensor: Output of shape (N, C, S * H, W)
        """

        return self.model(x, mask)

    def configure_optimizers(self):
        """
        Configure optimizers
        """

        return optim.Adam(
            self.parameters(), lr=0.001, betas=[0.9, 0.999], weight_decay=0.1
        )

    def training_step(self, batch: List[Tensor], batch_index: int):
        """
        Training step

        Note:
            N: batch size
            S: sequence length
            C: channel size
            H: image height
            W: image width

        Args:
            batch (List[Tensor]): list of S tensors of shape (N, C, H, W)
        """
        current_batch = cat(batch, dim=2)
        target = current_batch

        output = self(current_batch, self.mask)
        output = rshift_2d(output, self.patch_size)
        output[:, :, 0 : self.patch_size, 0 : self.patch_size] = target[
            :, :, 0 : self.patch_size, 0 : self.patch_size
        ]

        return self.loss(output, target)
