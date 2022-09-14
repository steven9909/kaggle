from genericpath import isdir
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from dataset import VideoDataModule
from model import Model
from torchvision import transforms as T

import os
from pathlib import Path


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((config.image_size, config.image_size)),
            T.Normalize(config.normalize.mean, config.normalize.std),
        ]
    )
    data_module = VideoDataModule(
        config.data_dir, config.batch_size, config.clip_len, transform
    )

    image_size = config.image_size
    patch_size = config.patch_size

    def get_latest_checkpoint(checkpoint_dir: Path):
        max_version_num = -1
        checkpoint_dir = checkpoint_dir / "lightning_logs"
        for file in checkpoint_dir.iterdir():
            print(file)
            if file.is_dir():
                max_version_num = max(max_version_num, int(file.stem.split("_")[-1]))

        if max_version_num == -1:
            return None

        checkpoint_dir = checkpoint_dir / f"version_{max_version_num}/checkpoints/"
        for file in checkpoint_dir.iterdir():
            if file.suffix == ".ckpt":
                return file

    model = Model(
        image_size,
        patch_size,
        config.in_channels,
        config.n_heads,
        config.d_token,
        config.n_enc_layers,
        config.n_dec_layers,
        config.clip_len * (image_size // patch_size) ** 2,
        config.clip_len,
    )
    trainer = pl.Trainer(
        default_root_dir=config.checkpoint_dir,
        max_epochs=5000,
        accelerator="gpu",
        resume_from_checkpoint=get_latest_checkpoint(Path(config.checkpoint_dir))
        if config.use_checkpoints
        else None,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
