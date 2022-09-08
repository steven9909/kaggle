import pytorch_lightning as pl
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from dataset import _VideoDataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        configuration = ViTConfig(
            hidden_size=64, intermediate_size=256, num_attention_heads=8
        )
        self.model = ViTModel(configuration)

    def forward(self, image: Tensor) -> Tensor:

        return self.model(image)["last_hidden_state"]


class Decoder(nn.Module):
    def __init__(self):

        super().__init__()
        # self.model = nn.TransformerDecoder()

    def forward(self, tensor: Tensor):
        pass


class Model(pl.LightningModule):
    def __init__(self):

        self.encoder = Encoder()


if __name__ == "__main__":
    dataset = _VideoDataset("./videos/", 16, lambda x: TF.to_tensor(x))
    x = DataLoader(dataset, batch_size=2)
    encoder = Encoder()
    for i in x:
        print(i.shape)
        break
