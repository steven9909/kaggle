import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataset import ImageDataset
from model.model import Model


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    model = Model(config.use_np_dataset)
    model.load_from_checkpoint(os.getcwd() / Path(config.checkpoint_dir) / "last.ckpt")
    model.eval()

    data = ImageDataset(os.getcwd() / Path(config.data_dir), batch_size=1)
    data.setup("")
    for image in data.train_dataloader():
        y, y_hat = model.forward(image)
        print(y)
        print(y_hat)
        break


if __name__ == "__main__":
    main()
