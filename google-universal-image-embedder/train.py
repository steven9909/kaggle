from pathlib import Path
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.dataset import Dataset

from model.model import Model

from pathlib import Path

import os


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    data_module = Dataset(
        os.getcwd() / Path(config.data_dir), batch_size=256, num_workers=10
    )

    ckpt_path = os.getcwd() / Path(config.checkpoint_dir)

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=2,
        monitor="loss",
        save_last=True,
    )

    ckpt_path = ckpt_path / "last.ckpt" if config.load_from_checkpoint else None

    model = Model()
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", callbacks=[checkpoint])
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
