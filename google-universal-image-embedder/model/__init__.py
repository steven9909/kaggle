import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms.functional as TF
from dataset import VideoDataModule
from torch import cat, from_numpy, optim, roll

from model.transformer.model import ViT


IMAGE_SIZE = 224
D_TOKEN = 64
N_ENC = 4
N_DEC = 4
HEADS = 8
FRAME_COUNT = 16
PATCH_SIZE = 16
SEQ_LEN = ((IMAGE_SIZE // PATCH_SIZE) ** 2) * FRAME_COUNT


def transform(x: np.ndarray):
    y = TF.to_tensor(x)
    y = TF.resize(y, [IMAGE_SIZE, IMAGE_SIZE])
    y = TF.normalize(y, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return y


def get_mask(seq_len: int):
    mask = np.tril(np.ones((1, seq_len, seq_len))).astype("uint8")
    mask = from_numpy(mask)
    return mask


class Model(pl.LightningModule):
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
        mask = get_mask(seq_len)
        self.register_buffer("mask", mask)

        self.model = ViT(
            d_token=d_token,
            heads=heads,
            n_encoder=n_encoder,
            n_decoder=n_decoder,
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=in_channels,
            image_size=image_size,
        )
        self.loss = nn.MSELoss()

    def forward(self, x, mask):
        return self.model(x, mask)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.002)

    def training_step(self, batch, batch_idx):
        current_batch = cat(batch, dim=2)
        target_batch = roll(current_batch, -1)

        output = self(current_batch, self.mask)
        output[:, :, -1, -1] = target_batch[:, :, -1, -1]

        return self.loss(output, target_batch)


if __name__ == "__main__":
    datamodule = VideoDataModule("./videos/", 16, transform=transform, batch_size=2)
    model = Model(
        d_token=D_TOKEN,
        heads=HEADS,
        n_encoder=N_ENC,
        n_decoder=N_DEC,
        seq_len=SEQ_LEN,
        patch_size=PATCH_SIZE,
        in_channels=3,
        image_size=IMAGE_SIZE,
    )
    trainer = pl.Trainer(max_epochs=2, accelerator="gpu")
    trainer.fit(model, datamodule=datamodule)
