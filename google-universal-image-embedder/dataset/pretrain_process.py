import random
import string
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import local

import torch
from torch import Tensor
from torchvision.models import VisionTransformer

from dataset.dataset import ImageDataset


class PretrainLoader:
    def __init__(
        self,
        image_dir: Path,
        save_dir: Path,
        pretrained_model: VisionTransformer,
        num_threads: int = 10,
    ):
        self.model = pretrained_model
        self.save_dir = save_dir
        self.dataset = ImageDataset(image_dir, batch_size=1)
        self.num_threads = num_threads
        random.seed(datetime.now())

    def _process_image(self, image: Tensor, local):
        image = local.model.forward(image)
        torch.save(image, self.save_dir / self.generate_filename())

    def process(self):
        def init_worker(local):
            model = deepcopy(self.model)
            model.eval()
            local.model = model

        thread_local = local()
        executor = ThreadPoolExecutor(
            initializer=init_worker,
            init_args=(thread_local),
            max_workers=self.num_threads,
        )

        for image in self.dataset.train_dataloader():
            executor.submit(self._process_image, image, thread_local)

    def generate_filename(self, N=30) -> str:
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=30))
