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
    )
    trainer = pl.Trainer(max_epochs=2, accelerator="gpu")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
