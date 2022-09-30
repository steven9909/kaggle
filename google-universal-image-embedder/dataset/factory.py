import json
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
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

    @staticmethod
    def get_stanford_cars_dataset(data_dir: Path) -> "StanfordCarsDataset":

        return StanfordCarsDataset(data_dir)

    @staticmethod
    def get_image_net_sketch_dataset(data_dir: Path) -> "ImageNetSketchDataset":

        return ImageNetSketchDataset(data_dir)


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

    def setup(self):

        raise NotImplementedError()

    def clean(self):

        raise NotImplementedError()


def move_all_sub_files_to_main(
    sub_folder_path: Path, main_folder_path: Path, remove_subfolder_path=True
):
    if not sub_folder_path.is_dir():
        return

    for file in sub_folder_path.iterdir():
        file.rename((main_folder_path / str(uuid.uuid4())).with_suffix(file.suffix))

    if remove_subfolder_path:
        sub_folder_path.rmdir()


class KaggleCompetition(Kaggle):
    def __init__(self, competition: str, data_dir: Path):
        super().__init__(competition, data_dir, "competition")


class KaggleDataset(Kaggle):
    def __init__(self, dataset: str, data_dir: Path):
        super().__init__(dataset, data_dir, "dataset")


class IMaterialistFashion2021FGVC8(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__("imaterialist-fashion-2021-fgvc8", data_dir)


class StanfordCarsDataset(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__("jessicali9530/stanford-cars-dataset", data_dir)

    def setup(self):

        move_all_sub_files_to_main(
            self.raw_data_dir / "cars_test/cars_test", self.raw_data_dir
        )
        move_all_sub_files_to_main(
            self.raw_data_dir / "cars_train/cars_train", self.raw_data_dir
        )

    def clean(self):

        (self.raw_data_dir / "cars_annos.mat").unlink()


class ImageNetSketchDataset(KaggleDataset):
    def __init__(self, data_dir: Path):

        super().__init__("wanghaohan/imagenetsketch", data_dir)

    def setup(self):
        pass


class IMaterialistChallengeFurniture2018(KaggleCompetition):
    def __init__(self, data_dir: Path):

        super().__init__("imaterialist-challenge-furniture-2018", data_dir)

    def _download_file(self, url: str, dir: Path):

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
                        self._download_file,
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
                        self._download_file,
                        image["url"][0],
                        valid_dir / str(image_to_annotation[image["image_id"]]),
                    ).add_done_callback(lambda _: progress.advance(valid_download))

            executor.shutdown(True)

    def clean(self):
        pass


if __name__ == "__main__":
    DatasetFactory.get_imaterialist_challenge_furniture_2018(Path("data/"))
