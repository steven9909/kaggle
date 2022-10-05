import zipfile
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Literal
from urllib.parse import urlparse

import kaggle
import requests
from PIL import Image, UnidentifiedImageError
from rich.progress import Progress


class Category(Enum):
    OTHER = 0
    APPAREL = 1
    ARTWORK = 2
    DISHES = 3
    STOREFRONTS = 4
    LANDMARKS = 5
    TOYS = 6
    PACKAGED_GOODS = 7
    FURNITURE = 8
    ALL = 9


def download_file(url: str, dir: Path):

    if not dir.exists():
        dir.mkdir(parents=True)

    with open(dir / Path(urlparse(url).path).name, "wb") as f:
        f.write(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content)


def download_files(urls: List[str], dir: Path):

    with ThreadPoolExecutor(16) as executor, Progress() as progress:
        tid = progress.add_task("Downloading", total=len(urls))

        for url in urls:
            future = executor.submit(download_file(url, dir))
            future.add_done_callback(lambda _: progress.advance(tid))

        executor.shutdown(True)


def iter_images(data_dir: Path) -> Iterator[Path]:

    for path in data_dir.rglob("*"):
        if path.is_file():
            try:
                Image.open(path)

                yield path

            except UnidentifiedImageError:
                continue


class DatasetFactory:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_kaggles(self, type: Category):

        if type == Category.ALL:
            return [
                IMaterialistFashion2020FGVC7(self.data_dir),
                StanfordCarsDataset(self.data_dir),
                ImageNetSketchDataset(self.data_dir),
                GuieToysDataset(self.data_dir),
                BestArtworksOfAllTime(self.data_dir),
                FoodRecognition2022(self.data_dir),
                FurnitureIdentificationDataset(self.data_dir),
            ]

        else:
            raise NotImplementedError


class Kaggle:
    data_zip: Path
    data_dir: Path
    samples: List[Path]

    def __init__(
        self, data_dir: Path, src: str, api: Literal["competition", "dataset"]
    ):

        name = src.split("/").pop()
        self.data_dir = data_dir / name
        self.data_zip = self.data_dir.with_suffix(".zip")

        if not self.data_zip.exists():
            if api == "competition":
                kaggle.api.competition_download_cli(src, path=data_dir)

            elif api == "dataset":
                kaggle.api.dataset_download_cli(src, path=data_dir)

            else:
                raise ValueError(f"unsupported api {api}")

        if not self.data_dir.exists():
            with zipfile.ZipFile(self.data_zip, "r") as z:
                z.extractall(self.data_dir)

    def iter_samples(self) -> Iterator[Path]:

        return iter_images(self.data_dir)


class KaggleCompetition(Kaggle):
    def __init__(self, data_dir: Path, competition: str):

        super().__init__(data_dir, competition, "competition")


class KaggleDataset(Kaggle):
    def __init__(self, data_dir: Path, dataset: str):

        super().__init__(data_dir, dataset, "dataset")


class IMaterialistFashion2020FGVC7(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__(data_dir, "imaterialist-fashion-2020-fgvc7")


class FoodRecognition2022(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__(data_dir, "sainikhileshreddy/food-recognition-2022")


class BestArtworksOfAllTime(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__(data_dir, "ikarus777/best-artworks-of-all-time")


class GuieToysDataset(KaggleDataset):
    def __init__(self, data_dir: Path):
        super().__init__(data_dir, "alejopaullier/guie-toys-dataset")


class StanfordCarsDataset(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__(data_dir, "jessicali9530/stanford-cars-dataset")


class ImageNetSketchDataset(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__(data_dir, "wanghaohan/imagenetsketch")


class FurnitureIdentificationDataset(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__(data_dir, "day-3-kaggle-competition")


if __name__ == "__main__":
    kaggles = DatasetFactory(Path("data/")).get_kaggles(Category.ALL)

    for path in kaggles[0].iter_samples():
        print(path)
