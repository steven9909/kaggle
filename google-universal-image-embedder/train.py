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
        config.data_dir, config.seq_len, config.batch_size, transform
    )
    model = Model(
        config.image_size,
        config.patch_size,
        config.in_channels,
        config.n_heads,
        config.d_token,
        config.n_enc_layers,
        config.n_dec_layers,
        config.seq_len,
    )
    trainer = pl.Trainer(max_epochs=2, accelerator="cpu")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
