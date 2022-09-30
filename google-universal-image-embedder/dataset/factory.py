import json
from typing import Callable, Any
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import kaggle
import requests
from rich.progress import Progress


class DatasetFactory:
    @staticmethod
    def get_imaterialist_fashion_2021_fgvc8(
        data_dir: Path,
    ) -> "IMaterialistFashion2021FGVC8":

        return IMaterialistFashion2021FGVC8(data_dir)

    @staticmethod
    def get_imaterialist_challenge_furniture_2018(
        data_dir: Path,
    ) -> "IMaterialistChallengeFurniture2018":

        return IMaterialistChallengeFurniture2018(data_dir)


class Kaggle:
    def __init__(
        self, url: str, data_dir: Path, api_callable: Callable[[str, Path], None]
    ):
        self.raw_data_zip = data_dir / (url + ".zip")
        self.raw_data_dir = data_dir / (url)

        if not self.raw_data_zip.exists():
            api_callable(url, path=data_dir)

        if not self.raw_data_dir.exists():
            with zipfile.ZipFile(self.raw_data_zip, "r") as z:
                z.extractall(self.raw_data_dir)

        self.setup()
        self.clean()

    def clean(self):
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()


class KaggleCompetition:
    def __init__(self, competition: str, data_dir: Path):
        super().__init__(competition, data_dir, kaggle.api.competition_download_cli)


class KaggleDataset:
    def __init__(self, dataset: str, data_dir: Path):
        super().__init__(dataset, data_dir, kaggle.api.dataset_download_cli)


class IMaterialistFashion2021FGVC8(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__("imaterialist-fashion-2021-fgvc8", data_dir)


class IMaterialistChallengeFurniture2018(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__("imaterialist-challenge-furniture-2018", data_dir)

    def download_file(self, url: str, dir: Path):
        if not dir.exists():
            dir.mkdir(parents=True)

        with open(dir / Path(urlparse(url).path).name, "wb") as f:
            f.write(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content)

    def setup(self):

        with ThreadPoolExecutor(16) as executor, Progress() as progress:
            with open(self.raw_data_dir / "train.json", "r") as f:
                train_dir = self.raw_data_dir / "train"
                train_metadata = json.load(f)
                images, annotations = (
                    train_metadata["images"],
                    train_metadata["annotations"],
                )
                train_download = progress.add_task(
                    "Downloading Train Data", total=len(images)
                )
                image_to_annotation = {
                    annotation["image_id"]: annotation["label_id"]
                    for annotation in annotations
                }

                for image in images:
                    executor.submit(
                        self.download_file,
                        image["url"][0],
                        train_dir / str(image_to_annotation[image["image_id"]]),
                    ).add_done_callback(lambda _: progress.advance(train_download))

            with open(self.raw_data_dir / "validation.json", "r") as f:
                valid_dir = self.raw_data_dir / "valid"
                valid_metadata = json.load(f)
                images, annotations = (
                    valid_metadata["images"],
                    valid_metadata["annotations"],
                )
                valid_download = progress.add_task(
                    "Downloading Valid Data", total=len(images)
                )
                image_to_annotation = {
                    annotation["image_id"]: annotation["label_id"]
                    for annotation in annotations
                }

                for image in images:
                    executor.submit(
                        self.download_file,
                        image["url"][0],
                        valid_dir / str(image_to_annotation[image["image_id"]]),
                    ).add_done_callback(lambda _: progress.advance(valid_download))

            executor.shutdown(True)

    def clean(self):
        pass


if __name__ == "__main__":
    DatasetFactory.get_imaterialist_challenge_furniture_2018(Path("data/"))
