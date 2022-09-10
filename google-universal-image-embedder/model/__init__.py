import pytorch_lightning as pl
from dataset import _VideoDataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from model.transformer.model import ViT
import numpy as np
import torch


def transform(x: np.ndarray):
    y = TF.to_tensor(x)
    y = TF.resize(y, [224, 224])
    y = TF.normalize(y, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return y


def get_mask(seq_len: int):
    mask = np.tril(np.ones((1, seq_len, seq_len))).astype("uint8")
    mask = torch.from_numpy(mask)
    return mask


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.seq_len = 3136
        mask = get_mask(self.seq_len)
        self.register_buffer("mask", mask)

        self.model = ViT(seq_len=self.seq_len)

    def forward(self, x, mask):
        return self.model(x, mask)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

    def training_step(self, batch, batch_idx):
        output = self(batch, self.mask)


if __name__ == "__main__":
    dataset = _VideoDataset("./videos/", 16, transform=transform)
    loader = DataLoader(dataset, batch_size=2)
    model = Model()
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu")
    trainer.fit(model, train_dataloaders=loader)
