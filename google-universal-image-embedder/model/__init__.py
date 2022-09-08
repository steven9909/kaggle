import pytorch_lightning as pl
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from transformers import ViTFeatureExtractor, ViTModel
from dataset import _VideoDataset
import torchvision.transforms.functional as TF


class Encoder(nn.Module):
    def __init__(self):

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


if __name__ == "__main__":
    dataset = _VideoDataset("./data/", 16, lambda x: TF.to_tensor(x))
    x = DataLoader(dataset, batch_size=2)

    for i in x:
        print(i.shape)
        break
