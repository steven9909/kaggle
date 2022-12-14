from torch import Tensor, cat, roll


def lshift_2d(x: Tensor, patch_size: int) -> Tensor:
    """
    Left shift 2D

    Note:
        N: batch size
        C: channel size
        H: image height
        W: image width

    Args:
        x (Tensor): Input tensor of shape (N, C, H, W)
        patch_size (int): Patch size

    Returns:
        Tensor: output of shape (N, C, H, W)
    """

    before, last_col = x[..., :-patch_size], x[..., -patch_size:]
    last_col = roll(last_col, patch_size, 2)

    return cat([last_col, before], 3)


def rshift_2d(x: Tensor, patch_size: int) -> Tensor:
    """
    Right shift 2D

    Note:
        N: batch size
        C: channel size
        H: image height
        W: image width

    Args:
        x (Tensor): Input tensor of shape (N, C, H, W)
        patch_size (int): Patch size

    Returns:
        Tensor: output of shape (N, C, H, W)
    """

    first_col, after = x[..., :patch_size], x[..., patch_size:]
    first_col = roll(first_col, -patch_size, 2)

    return cat([after, first_col], 3)
