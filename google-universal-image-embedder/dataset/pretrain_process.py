from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock, RLock
from pathlib import Path
from torchvision.models import VisionTransformer
from typing import Iterator, Optional, Callable
from torch import Tensor
from PIL import Image
import torch
from dataset.dataset import Dataset
import string
import random
from datetime import datetime


class PretrainLoader:
    def __init__(
        self,
        image_dir: Path,
        save_dir: Path,
        pretrained_model: VisionTransformer,
        num_threads: int = 0,
    ):
        self.model = pretrained_model
        self.model.eval()
        self.save_dir = save_dir
        self.dataset = Dataset(image_dir, batch_size=1)
        self.num_threads = num_threads
        random.seed(datetime.now())

    def _process_image(self, image: Tensor):
        image = self.model.forward(image)
        torch.save(image, self.save_dir / self.generate_filename())

    def process(self):
        executor = ThreadPoolExecutor(max_workers=self.num_threads)
        for image in self.dataset.train_dataloader():
            executor.submit(self._process_image, image)

    def generate_filename(self, N=30) -> str:
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=30))
