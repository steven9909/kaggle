from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional
from uuid import uuid4

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch import Tensor, from_numpy
from torch.utils import data
from torchvision import transforms as T


class ImageFolder(data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ):

        self.samples = list(data_dir.glob("*.jpeg"))
        self.transform = T.ToTensor() if transform is None else transform

    def __getitem__(self, index: int) -> Tensor:

        return self.transform(Image.open(self.samples[index]).convert("RGB"))

    def __len__(self) -> int:

        return len(self.samples)


class NumpyFolder(data.Dataset):
    def __init__(
        self,
        data_dir: Path,
    ):

        self.samples = list(data_dir.glob("*.npy"))

    def __getitem__(self, index: int) -> Tensor:

        return from_numpy(np.load(self.samples[index]))

    def __len__(self) -> int:

        return len(self.samples)

    @staticmethod
    def convert_from_image(
        src_dir: Path,
        dst_dir: Path,
        conversion_fn: Callable[[List[Path]], Tensor],
        batch_size: int = 64,
        max_workers: int = 8,
    ):

        dst_dir.mkdir(parents=True, exist_ok=True)

        def done_callback(x: Future[Tensor]):
            x_ = x.result()

            for i in range(batch_size):
                np.save(dst_dir / f"{uuid4()}", x_[i].numpy())

        with ThreadPoolExecutor(max_workers) as executor:
            samples = []

            for sample in src_dir.glob("*.JPEG"):
                samples.append(sample)

                if len(samples) == batch_size:
                    future = executor.submit(conversion_fn, samples.copy())
                    future.add_done_callback(done_callback)
                    samples.clear()

            if len(samples) > 0:
                future = executor.submit(conversion_fn, samples.copy())
                future.add_done_callback(done_callback)
                samples.clear()

            executor.shutdown(True)


class ImageDataset(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 64, num_workers: int = 0):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(224),
                T.CenterCrop(224),
                T.Normalize([0.485, 0.456, 0.406], (0.229, 0.224, 0.225)),
            ]
        )

    def setup(self, stage: str):

        dataset = ImageFolder(self.data_dir, self.transform)

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


class NumpyDataset(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 64, num_workers: int = 0):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):

        dataset = NumpyFolder(self.data_dir)

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
