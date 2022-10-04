from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch import Tensor, from_numpy
from torch.utils import data
from torchvision import transforms as T

from dataset.factory import Extension, Kaggle
from itertools import chain


class Contrastive(data.Dataset):
    def __init__(
        self,
        kaggles: List[Kaggle],
        extension: Extension,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ):

        self.samples = list(
            chain([kaggle.iter_samples(extension) for kaggle in kaggles])
        )
        self.transform = T.ToTensor() if transform is None else transform

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        image = Image.open(self.samples[index]).convert("RGB")

        return (
            self.transform(image),
            self.transform(image),
        )

    def __len__(self) -> int:

        return len(self.samples)


class BYOLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        kaggles: List[Kaggle],
        extension: Extension,
        batch_size: int = 64,
        num_workers: int = 0,
    ):

        super().__init__()
        self.kaggles = kaggles
        self.extension = extension
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = (
            T.Compose(
                [
                    T.ToTensor(),
                    T.RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
                    T.RandomGrayscale(p=0.2),
                    T.RandomHorizontalFlip(),
                    T.RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                    T.RandomResizedCrop((224, 224)),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )

    def setup(self, stage: str):

        dataset = Contrastive(self.kaggles, self.extension, self.transform)

        fit_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        tst_len = len(dataset) - fit_len - val_len

        self.datasets = data.random_split(dataset, [fit_len, val_len, tst_len])

    def train_dataloader(self):
        return data.DataLoader(
            self.datasets[0], batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.datasets[1], batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.datasets[2], batch_size=self.batch_size, num_workers=self.num_workers
        )
