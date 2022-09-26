from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch import jit

from model import Model


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    model = Model.load_from_checkpoint(Path(config.checkpoint_dir, "last-v2.ckpt"))
    m = jit.script(model.model)
    jit.save(m, "submission.pt")

    from PIL import Image
    from torchvision.transforms import functional as TF

    m = jit.load("submission.pt")
    m = m.eval()
    image = Image.open("C:/Users/JBenn/Downloads/NebraskaImage_small.jpg")

    print(m(TF.pil_to_tensor(image).unsqueeze(0)).shape)


if __name__ == "__main__":
    main()
