from dataset.factory import DatasetFactory, DatasetType
from pathlib import Path

factory = DatasetFactory(Path("data/"))
[func() for func in factory.get_dataset_func(DatasetType.ALL)]
