import copy
import math
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, drop_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.heads = heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(drop_p)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_lin(q).view(batch_size, -1, self.heads, self.d_k)
        k = self.k_lin(k).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_lin(v).view(batch_size, -1, self.heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = torch.nn.LayerNorm(d_model)
        self.norm_2 = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadedAttention(heads, d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, seq_len):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, seq_len)
        self.layers = get_clones(AttentionLayer(d_model, heads), N)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(AttentionLayer(d_model, heads), N)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


# Perform patch embedding using a convolution layer
class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, in_channels, d_token):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, d_token, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # B, S, C, H, W = x.shape
        # switch channel and sequence dimension to preserve channel when we reshape later
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(B, C, S * H, W)
        x = self.proj(x)

        # after projection x: (B, D_TOKEN, (IMAGE_SIZE*SEQ_LEN)/PATCH_SIZE, IMAGE_SIZE/PATCH_SIZE)
        x = x.flatten(2).transpose(1, 2)
        # final shape of x: (B, (IMAGE_SIZE^2*SEQ_LEN)/PATCH_SIZE^2, D_TOKEN)
        return x


# Perform patch deembedding using a deconvolution layer
# Eseentially the reverse operation of PatchEmbedder
class PatchUnembedder(nn.Module):
    def __init__(self, d_token, out_channels, patch_size, image_size):
        super().__init__()
        self.unproj = nn.ConvTranspose2d(
            d_token, out_channels, kernel_size=patch_size, stride=patch_size
        )
        self.image_size = image_size
        self.patch_size = patch_size

    # input x shape: (B, (IMAGE_SIZE^2*SEQ_LEN)/PATCH_SIZE^2, D_TOKEN)
    def forward(self, x: Tensor):
        print(x.shape)
        x = x.unflatten(
            1, (self.image_size, self.image_size // self.patch_size)
        ).transpose(1, 2)
        print(x.shape)
        x = self.unproj(x)
        # B, C, S_H, W = x.shape
        # x = x.view(B, C, S_H // self.image_size, self.image_size, W)
        # x = x.transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        d_token,
        n_encoder,
        n_decoder,
        heads,
        seq_len,
        patch_size,
        in_channels,
        image_size,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedder(
            patch_size=patch_size,
            in_channels=in_channels,
            d_token=d_token,
        )

        self.patch_deembed = PatchUnembedder(
            patch_size=patch_size,
            d_token=d_token,
            out_channels=in_channels,
            image_size=image_size,
        )

        self.encoder = Encoder(
            d_model=d_token, N=n_encoder, heads=heads, seq_len=seq_len
        )
        self.decoder = Decoder(d_model=d_token, N=n_decoder, heads=heads)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        x = self.encoder(x, mask)
        x = self.decoder(x, mask)
        x = self.patch_deembed(x)
        return x
