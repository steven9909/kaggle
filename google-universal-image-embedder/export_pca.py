from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch import jit

from model import PCAModel


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    model = PCAModel(Path("pca.pkl"))
    m = jit.script(model)
    jit.save(m, "saved_model.pt")
    m = jit.load("saved_model.pt")

    from PIL import Image
    from torchvision.transforms import functional as TF

    model = PCAModel(Path("pca.pkl")).eval()
    image = Image.open("C:/Users/JBenn/Downloads/NebraskaImage_small.jpg")
    print(model(TF.pil_to_tensor(image).unsqueeze(0)))


if __name__ == "__main__":
    main()
