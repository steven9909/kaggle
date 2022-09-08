from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import pytorch_lightning as pl
from torch import nn
from torch import Tensor
from PIL.Image import Image as PILImage


class Encoder(nn.Module):
    def __init__(self, pretrained: str = "google/vit-base-patch16-224-in21k"):

        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained)
        self.model = ViTModel.from_pretrained(pretrained)

    def forward(self, image: PILImage) -> Tensor:

        return self.model(**self.feature_extractor(image, return_tensors="pt"))[
            "last_hidden_state"
        ]


class Decoder(nn.Module):
    def __init__(self):

        super().__init__()
        # self.model = nn.TransformerDecoder()

    def forward(self, tensor: Tensor):

        pass


class Model(pl.LightningModule):
    def __init__(self):

        self.encoder = Encoder()
        self.decoder = Decoder()
