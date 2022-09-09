import pytorch_lightning as pl
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from dataset import _VideoDataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from model.transformer.model import PatchEmbedder, ViT
import numpy as np


class Model(pl.LightningModule):
    def __init__(self):
        pass


def transform(x: np.ndarray):
    y = TF.to_tensor(x)
    y = TF.resize(y, [224, 224])
    y = TF.normalize(y, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return y


if __name__ == "__main__":
    dataset = _VideoDataset("./videos/", 16, transform=transform)
    x = DataLoader(dataset, batch_size=2)
    encoder = ViT()
    for i in x:
        print(i.shape)
        output = encoder(i, None)
        print(output.shape)
        break
