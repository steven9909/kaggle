from typing import List

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from torch import BoolTensor, Tensor, cat, optim, roll

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
        mask = construct_mask(seq_len // n_heads)
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

        return optim.Adam(self.parameters(), lr=0.002)

    def training_step(self, batch: List[Tensor], _):
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

        # TODO: needs rework
        current_batch = cat(batch, dim=2)
        target_batch = roll(current_batch, -1)

        output = self(current_batch, self.mask)
        output[:, :, -1, -1] = target_batch[:, :, -1, -1]
        # TODO: needs rework

        return self.loss(output, target_batch)
