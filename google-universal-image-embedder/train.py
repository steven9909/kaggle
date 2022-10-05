from pathlib import Path

import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.factory_loader import BYOLDataModule
from dataset.factory import Category, DatasetFactory

from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from model import BYOLLightningModule

from torch.nn import Identity

import pytorch_lightning as pl

from torch import Tensor, randn


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    kaggles = DatasetFactory(Path(config.data_dir)).get_kaggles(Category.ALL)
    data_module = BYOLDataModule(
        kaggles, batch_size=config.batch_size, num_workers=10, n=20000
    )

    ckpt_path = Path(config.checkpoint_dir)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=-1,
        monitor="loss",
        save_last=True,
    )
    ckpt_path = ckpt_path / "last.ckpt" if config.load_from_checkpoint else None

    repr_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

    for name, param in repr_model.named_parameters():
        if not (
            name.startswith("features.6")
            or name.startswith("features.7")
            or name.startswith("classifier")
        ):
            param.requires_grad = False

    repr_model.classifier[2] = Identity()

    model = BYOLLightningModule(repr_model, 1024, 64, 4096)
    trainer = pl.Trainer(
        max_epochs=500, accelerator="gpu", devices=1, callbacks=[checkpoint]
    )
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
