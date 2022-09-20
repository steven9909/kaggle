import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import ImageDataset, NumpyDataset
from model.model import Model


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    data_module = (
        NumpyDataset(
            os.getcwd() / Path(config.data_dir), batch_size=256, num_workers=10
        )
        if config.use_np_dataset
        else ImageDataset(
            os.getcwd() / Path(config.data_dir), batch_size=256, num_workers=10
        )
    )

    ckpt_path = os.getcwd() / Path(config.checkpoint_dir)

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=-1,
        monitor="loss",
        save_last=True,
    )

    ckpt_path = ckpt_path / "last.ckpt" if config.load_from_checkpoint else None

    model = Model(config.use_np_dataset)
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", callbacks=[checkpoint])
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
