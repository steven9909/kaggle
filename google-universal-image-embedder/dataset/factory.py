import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Literal
from urllib.parse import urlparse
from uuid import uuid4

import kaggle
import requests
from rich.progress import Progress


class Extension(str, Enum):
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"


def download_file(url: str, dir: Path):

    if not dir.exists():
        dir.mkdir(parents=True)

    with open(dir / Path(urlparse(url).path).name, "wb") as f:
        f.write(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content)


def download_files(urls: List[str], dir: Path):

    with ThreadPoolExecutor(16) as executor, Progress() as progress:
        tid = progress.add_task("Downloading", total=len(urls))

        for url in urls:
            executor.submit(download_file(url, dir)).add_done_callback(
                lambda _: progress.advance(tid)
            )

        executor.shutdown(True)


def rglob2root(glob: Path, root: Path, extension: Extension, remove: bool = False):
    if not glob.is_dir():
        return

    for path in glob.rglob(f"*{extension}"):
        path.rename(root / f"{uuid4()}{extension}")

    if remove:
        shutil.rmtree(glob)


class DatasetFactory:
    @staticmethod
    def get_imaterialist_fashion_2020_fgvc7(
        data_dir: Path,
    ) -> "IMaterialistFashion2020FGVC7":

        return IMaterialistFashion2020FGVC7(data_dir)

    @staticmethod
    def get_imaterialist_challenge_furniture_2018(
        data_dir: Path,
    ) -> "IMaterialistChallengeFurniture2018":

        return IMaterialistChallengeFurniture2018(data_dir)

    @staticmethod
    def get_stanford_cars_dataset(data_dir: Path) -> "StanfordCarsDataset":

        return StanfordCarsDataset(data_dir)

    @staticmethod
    def get_image_net_sketch_dataset(data_dir: Path) -> "ImageNetSketchDataset":

        return ImageNetSketchDataset(data_dir)

    @staticmethod
    def get_guie_toys_dataset(data_dir: Path) -> "GuieToysDataset":

        return GuieToysDataset(data_dir)

    @staticmethod
    def get_best_artworks_of_all_time(data_dir: Path) -> "BestArtworksOfAllTime":

        return BestArtworksOfAllTime(data_dir)

    @staticmethod
    def get_food_recognition_2022(data_dir: Path) -> "FoodRecognition2022":

        return FoodRecognition2022(data_dir)


class Kaggle:
    download_cli_factory = {
        "competition": kaggle.api.competition_download_cli,
        "dataset": kaggle.api.dataset_download_cli,
    }

    def __init__(
        self, src: str, data_dir: Path, api: Literal["competition", "dataset"]
    ):

        name = src.split("/")[-1]
        self.raw_data_zip = data_dir / (name + ".zip")
        self.raw_data_dir = data_dir / (name)

        if not self.raw_data_zip.exists():
            self.download_cli_factory[api](src, path=data_dir)

        if not self.raw_data_dir.exists():
            with zipfile.ZipFile(self.raw_data_zip, "r") as z:
                z.extractall(self.raw_data_dir)

        self.setup()
        self.clean()

    def iter_samples(self, extension: Extension) -> Iterator[Path]:

        return self.raw_data_dir.glob(f"*{extension}")

    def setup(self):

        raise NotImplementedError()

    def clean(self):

        raise NotImplementedError()


class KaggleCompetition(Kaggle):
    def __init__(self, competition: str, data_dir: Path):

        super().__init__(competition, data_dir, "competition")


class KaggleDataset(Kaggle):
    def __init__(self, dataset: str, data_dir: Path):

        super().__init__(dataset, data_dir, "dataset")


class IMaterialistFashion2020FGVC7(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__("imaterialist-fashion-2020-fgvc7", data_dir)

    def setup(self):

        rglob2root(self.raw_data_dir / "train", self.raw_data_dir, Extension.JPG, True)
        rglob2root(self.raw_data_dir / "test", self.raw_data_dir, Extension.JPG, True)

    def clean(self):

        os.remove(self.raw_data_dir / "sample_submission.csv")
        os.remove(self.raw_data_dir / "train.csv")


class FoodRecognition2022(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__("sainikhileshreddy/food-recognition-2022", data_dir)

    def setup(self):

        rglob2root(
            self.raw_data_dir / "raw_data", self.raw_data_dir, Extension.JPG, True
        )

    def clean(self):

        os.remove(self.raw_data_dir / "visualize_dataset.png")
        shutil.rmtree(self.raw_data_dir / "hub", ignore_errors=True)


class BestArtworksOfAllTime(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__("ikarus777/best-artworks-of-all-time", data_dir)

    def setup(self):

        rglob2root(
            self.raw_data_dir / "resized", self.raw_data_dir, Extension.JPG, True
        )

    def clean(self):

        os.remove(self.raw_data_dir / "artists.csv")
        shutil.rmtree(self.raw_data_dir / "images", ignore_errors=True)


class GuieToysDataset(KaggleDataset):
    def __init__(self, data_dir: Path):
        super().__init__("alejopaullier/guie-toys-dataset", data_dir)

    def setup(self):

        rglob2root(self.raw_data_dir / "toys", self.raw_data_dir, Extension.JPG, True)

    def clean(self):

        (self.raw_data_dir / "toys.csv").unlink()


class StanfordCarsDataset(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__("jessicali9530/stanford-cars-dataset", data_dir)

    def setup(self):

        rglob2root(
            self.raw_data_dir / "cars_test/", self.raw_data_dir, Extension.JPG, True
        )
        rglob2root(
            self.raw_data_dir / "cars_train/", self.raw_data_dir, Extension.JPG, True
        )

    def clean(self):

        (self.raw_data_dir / "cars_annos.mat").unlink()


class ImageNetSketchDataset(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__("wanghaohan/imagenetsketch", data_dir)

    def setup(self):

        rglob2root(self.raw_data_dir, self.raw_data_dir, Extension.JPEG)


class IMaterialistChallengeFurniture2018(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__("imaterialist-challenge-furniture-2018", data_dir)

    def setup(self):

        with open(self.raw_data_dir / "train.json", "r") as f:
            download_files(
                [image["url"][0] for image in json.load(f)["images"]], self.raw_data_dir
            )

        with open(self.raw_data_dir / "validation.json", "r") as f:
            download_files(
                [image["url"][0] for image in json.load(f)["images"]], self.raw_data_dir
            )

        with open(self.raw_data_dir / "test.json", "r") as f:
            download_files(
                [image["url"][0] for image in json.load(f)["images"]], self.raw_data_dir
            )

    def clean(self):
        pass


if __name__ == "__main__":
    DatasetFactory.get_imaterialist_fashion_2020_fgvc7(Path("data/"))
