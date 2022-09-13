from torch import Tensor, cat, roll
from torchvision.transforms import functional as TF


def lshift_2d(x: Tensor, patch_size: int) -> Tensor:
    """
    Left shift 2D

    Note:
        N: batch size
        S: sequence length
        C: channel size
        H: image height
        W: image width

    Args:
        x (Tensor): Input tensor of shape (N, S, C, H, W)
        patch_size (int): Patch size

    Returns:
        Tensor: output of shape (N, S, C, H, W)
    """

    before, last_col = x[..., :-patch_size], x[..., -patch_size:]
    last_col = roll(last_col, patch_size, 3)

    return cat([last_col, before], 4)


def rshift_2d(x: Tensor, patch_size: int) -> Tensor:
    """
    Right shift 2D

    Note:
        N: batch size
        S: sequence length
        C: channel size
        H: image height
        W: image width

    Args:
        x (Tensor): Input tensor of shape (N, S, C, H, W)
        patch_size (int): Patch size

    Returns:
        Tensor: output of shape (N, S, C, H, W)
    """

    first_col, after = x[..., :patch_size], x[..., patch_size:]
    first_col = roll(first_col, -patch_size, 3)

    return cat([after, first_col], 4)
