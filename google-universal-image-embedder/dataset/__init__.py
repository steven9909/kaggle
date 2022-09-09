import random
from pathlib import Path
from typing import Callable, Iterator, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from torch import Tensor, stack
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.folder import make_dataset


class Video:
    def __init__(self, path: Path):

        self.video = cv2.VideoCapture(path)

    def read(self, start: int, n: int) -> Iterator[np.ndarray]:

        self.video.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(n):
            yield self.video.read()[1]

    @property
    def length(self) -> int:

        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def shape(self) -> Tuple[int, int]:

        return (
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )


class _VideoDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        clip_len: int,
        transform: Callable[[np.ndarray], Tensor] = lambda x: TF.to_tensor(x),
    ):

        super(VideoDataset).__init__()
        self.data_dir = data_dir
        self.clip_len = clip_len
        self.transform = transform
        self.samples = make_dataset(data_dir, extensions="mp4")

    def __getitem__(self, idx: int) -> Tensor:

        video = Video(self.samples[idx][0])
        start = int(random.uniform(0.0, video.length - self.clip_len))
        frames = map(self.transform, video.read(start, self.clip_len))

        return stack(list(frames))

    def __len__(self) -> int:
        return len(self.samples)


class VideoDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        clip_len: int,
        transform: Callable[[np.ndarray], Tensor] = lambda x: TF.to_tensor(x),
        batch_size: int = 32,
    ):

        super().__init__()
        self.data_dir = data_dir
        self.clip_len = clip_len
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage: str):

        dataset = _VideoDataset(self.data_dir, self.clip_len, self.transform)

        fit_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        tst_len = len(dataset) - fit_len - val_len

        self.datasets = random_split(dataset, [fit_len, val_len, tst_len])

    def train_dataloader(self) -> DataLoader:

        return DataLoader(self.datasets[0], batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:

        return DataLoader(self.datasets[1], batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:

        return DataLoader(self.datasets[2], batch_size=self.batch_size)


def transform(x: np.ndarray):
    y = TF.to_tensor(x)
    y = TF.resize(y, [224, 224])
    y = TF.normalize(y, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return y


if __name__ == "__main__":
    dataset = _VideoDataset("./videos/", 16, transform=transform)
    x = DataLoader(dataset, batch_size=1)

    for i in x:
        print(i.shape)
        u = i.view(3, 3584, 224)
        TF.to_pil_image(u).save("saved.png")
        print(u.shape)
        break
