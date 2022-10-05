from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch import device, jit, load, nn
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

from model import BYOLLightningModule


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    repr_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    repr_model.classifier[2] = nn.Identity()

    model = BYOLLightningModule(repr_model, 1024, 64, 4096)
    model.load_state_dict(
        load(Path(config.checkpoint_dir, "epoch_3.ckpt"), map_location=device("cpu"))[
            "state_dict"
        ]
    )
    m = jit.script(model.model)
    jit.save(m, "saved_model.pt")

    from PIL import Image
    from torchvision.transforms import functional as TF

    m = jit.load("saved_model.pt")
    m = m.eval()
    image = Image.open("C:/Users/JBenn/Downloads/NebraskaImage_small.jpg")

    print(m(TF.pil_to_tensor(image).unsqueeze(0)).shape)


if __name__ == "__main__":
    main()
