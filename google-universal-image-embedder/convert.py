from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch import nn, no_grad
from torchvision import models
from torchvision import transforms as T

from dataset import NumpyFolder


def get_conversion_fn() -> Callable[[Path], np.ndarray]:

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize([0.485, 0.456, 0.406], (0.229, 0.224, 0.225)),
        ]
    )

    vit = models.vit_l_16(weights=models.ViT_L_16_Weights)
    vit.heads = nn.Identity()
    vit = vit.eval()

    def conversion_fn(sample: Path) -> np.ndarray:
        with no_grad():
            return (
                vit(transform(Image.open(sample).convert("RGB")).unsqueeze(0))
                .squeeze(0)
                .numpy(),
            )

    return conversion_fn


if __name__ == "__main__":
    NumpyFolder.convert_from_image(
        Path("data/imagenet_test"),
        Path("data/numpynet_test"),
        get_conversion_fn(),
        max_workers=16,
    )
