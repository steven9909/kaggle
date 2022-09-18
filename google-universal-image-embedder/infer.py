import hydra
from omegaconf import DictConfig
from model.model import Model
from dataset.dataset import Dataset
import os
from pathlib import Path
from torchvision.transforms import functional as TF


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    model = Model()
    model.load_from_checkpoint(os.getcwd() / Path(config.checkpoint_dir) / "last.ckpt")

    data = Dataset(os.getcwd() / Path(config.data_dir), batch_size=1)
    data.setup("")
    for image in data.train_dataloader():
        y, y_hat = model.forward(image)
        print(y)
        print(y_hat)
        break


if __name__ == "__main__":
    main()
