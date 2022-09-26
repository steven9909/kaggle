from pathlib import Path

import hydra
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import stack

from dataset.dataset import NumpyFolder
from model import Model


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    model = Model.load_from_checkpoint(Path(config.checkpoint_dir, "last-v2.ckpt"))
    dataset = NumpyFolder(Path(config.data_dir))

    sample = dataset[1]

    y_hat = model.forward(stack([sample, sample]))
    print(sample)
    print(y_hat[0])
    print(F.mse_loss(y_hat, sample))


if __name__ == "__main__":
    main()
