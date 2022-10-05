from itertools import chain, islice
from typing import Callable, List, Optional, Tuple, Iterable

import pytorch_lightning as pl
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils import data
from torchvision import transforms as T

import numpy as np

from dataset.factory import Extension, Kaggle


def _get_length_of_iterable(iterable: Iterable):
    return sum(1 for _ in iterable)


class Contrastive(data.Dataset):
    def __init__(
        self,
        kaggles: List[Kaggle],
        extensions: List[Extension],
        n: Optional[int] = None,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ):
        self.samples = [kaggle.iter_samples(extensions) for kaggle in kaggles]
        assert len(self.samples >= 1), "List of kaggle objects cannot be empty"

        if not n:
            n = _get_length_of_iterable(self.samples[0])
            for sample in self.samples[1:]:
                n = min(n, _get_length_of_iterable(sample))

        self.samples = chain(*[islice(sample, n) for sample in self.samples])
        self.transform = T.ToTensor() if transform is None else transform

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        try_index = index
        while True:
            try:
                with Image.open(self.samples[try_index]) as img:
                    img = img.convert("RGB")
                    return (
                        self.transform(img),
                        self.transform(img),
                    )
            except UnidentifiedImageError:
                try_index = np.random.randint(0, len(self.samples))
                continue

    def __len__(self) -> int:

        return len(self.samples)


class BYOLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        kaggles: List[Kaggle],
        extensions: List[Extension],
        batch_size: int = 64,
        num_workers: int = 0,
    ):

        super().__init__()
        self.kaggles = kaggles
        self.extensions = extensions
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
                T.RandomResizedCrop((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, _: str):

        dataset = Contrastive(self.kaggles, self.extensions, self.transform)

        fit_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        tst_len = len(dataset) - fit_len - val_len

        self.datasets = data.random_split(dataset, [fit_len, val_len, tst_len])

    def train_dataloader(self):
        return data.DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
