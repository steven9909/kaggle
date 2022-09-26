from pathlib import Path
from typing import Callable, List

from PIL import Image
from torch import Tensor, nn, stack
from torchvision import models
from torchvision import transforms as T

from dataset import NumpyFolder


def get_conversion_fn() -> Callable[[List[Path]], Tensor]:

    transform = T.Compose(
        [
            T.PILToTensor(),
            T.Resize([224, 224]),
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    vit = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)

    for parameter in vit.parameters():
        parameter.requires_grad = False

    vit.heads = nn.Identity()
    vit = vit.cuda()
    vit = vit.eval()

    def conversion_fn(samples: List[Path]) -> Tensor:
        images = [transform(Image.open(sample).convert("RGB")) for sample in samples]

        return vit(stack(images).cuda()).cpu()

    return conversion_fn


if __name__ == "__main__":
    NumpyFolder.convert_from_image(
        Path("data/imagenet_test"),
        Path("data/numpynet_test"),
        get_conversion_fn(),
        max_workers=16,
    )
