import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from dataset import VideoDataModule
from model import Model
from torchvision import transforms as T


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
        default_root_dir=config.checkpoint_dir, max_epochs=100, accelerator="gpu"
    )
    trainer.fit(
        model,
        data_module,
        ckpt_path=config.checkpoint_dir if config.use_checkpoints else None,
    )


if __name__ == "__main__":
    main()
