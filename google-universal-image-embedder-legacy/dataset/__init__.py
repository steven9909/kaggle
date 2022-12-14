import random
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.folder import make_dataset


class Video:
    def __init__(self, path: Path):
        self.video = cv2.VideoCapture(path)

    def read(self, start: int, n: int) -> Iterator[np.ndarray]:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(n):
            yield cv2.cvtColor(self.video.read()[1], cv2.COLOR_BGR2RGB)

    @property
    def length(self) -> int:
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def shape(self) -> Tuple[int, int]:
        return (
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )


class VideoDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        clip_len: int,
        transform: Callable[[np.ndarray], Tensor] = lambda x: TF.to_tensor(x),
    ):

        super().__init__()
        self.data_dir = data_dir
        self.clip_len = clip_len
        self.transform = transform
        self.samples = make_dataset(data_dir, extensions="mp4")

    def __getitem__(self, idx: int) -> List[Tensor]:

        video = Video(self.samples[idx][0])
        start = random.randrange(0.0, video.length - self.clip_len, 1)
        frames = map(self.transform, video.read(start, self.clip_len))

        return list(frames)

    def __len__(self) -> int:
        return len(self.samples)


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        clip_len: int = 16,
        transform: Callable[[np.ndarray], Tensor] = lambda x: TF.to_tensor(x),
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.clip_len = clip_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        dataset = VideoDataset(self.data_dir, self.clip_len, self.transform)

        fit_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        tst_len = len(dataset) - fit_len - val_len

        self.datasets = random_split(dataset, [fit_len, val_len, tst_len])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets[0], batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets[1], batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets[2], batch_size=self.batch_size, num_workers=self.num_workers
        )
