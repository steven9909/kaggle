from typing import List
import hydra
from omegaconf import DictConfig
from torch import Tensor
from torchvision import transforms as T
from PIL import Image

from model import Model


def forward_autoregress(model: Model, x: Tensor, num_frames: int) -> Tensor:
    """
    Forward autoregressive given model

    Note:
        N: batch size
        S: sequence length
        C: channel size
        H: image height
        W: image width

    Args:
        model (Model): Model for computing the features
        x (Tensor): Input tensor of shape (N, C, S * H, W)
        num_frames (int): Number of frames to generate

    Returns:
        Tensor: Output of shape (N, d_token)
    """

    for i in range(num_frames):
        pass


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((config.image_size, config.image_size)),
            T.Normalize(config.normalize.mean, config.normalize.std),
        ]
    )

    image_size = config.image_size
    patch_size = config.patch_size
    n_patchs_per_frame = (image_size // patch_size) ** 2

    model = Model(
        image_size,
        patch_size,
        config.in_channels,
        config.n_heads,
        config.d_token,
        config.n_enc_layers,
        config.n_dec_layers,
        config.clip_len * n_patchs_per_frame,
    )

    print(model.forward_features(transform(Image.open("tesla.png")).unsqueeze(0)).shape)


if __name__ == "__main__":
    main()
