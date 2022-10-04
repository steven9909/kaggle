from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
import pytorch_lightning as pl


class SimCLRLoss(nn.Module):
    def __init__(self, batch_size: int, simclr_temperature: float):

        super().__init__()

        self.batch_size = batch_size
        self.simclr_temperature = simclr_temperature

    def _cosine_similarity_matrix(self, z: Tensor) -> Tensor:

        return F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim = self._cosine_similarity_matrix(torch.cat([z1, z2], 0))
        exp = torch.exp(sim / self.simclr_temperature)

        mask = torch.eye(2 * self.batch_size).bool()
        pmask = mask.roll(self.batch_size, 0)
        nmask = ~mask

        return torch.mean(
            -torch.log(torch.sum(pmask * exp, 0) / torch.sum(nmask * exp, 1))
        )


class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q1: Tensor, q2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
        """
        q1: batch of N augumented (t), predicted embeddings (online)
        q2: batch of N augumented (t') predicted embeddings (online)
        z1: batch of N augumented (t') projected embeddings (target)
        z2: batch of N augumented (t) projected embeddings (target)
        """
        z1 = z1.detach()
        z2 = z2.detach()

        l_1 = torch.sum((q1 * z1), dim=-1) / (
            torch.norm(q1, dim=1, p=2) * torch.norm(z1, dim=1, p=2)
        )

        l_2 = torch.sum((q2 * z2), dim=-1) / (
            torch.norm(q2, dim=1, p=2) * torch.norm(z2, dim=1, p=2)
        )

        return 2 - 2 * torch.mean((l_1 + l_2))


class VICRegLoss(nn.Module):
    def __init__(
        self,
        batch_size: int,
        repr_size: int,
        vicreg_lambda: float,
        vicreg_mu: float,
        vicreg_nu: float,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.repr_size = repr_size
        self.vicreg_lambda = vicreg_lambda
        self.vicreg_mu = vicreg_mu
        self.vicreg_nu = vicreg_nu

    def _invariance(self, z1: Tensor, z2: Tensor) -> Tensor:

        return F.mse_loss(z1, z2)

    def _variance(self, z1: Tensor, z2: Tensor) -> Tensor:

        std_z1 = torch.sqrt(torch.var(z1, 0) + 1e-4)
        std_z2 = torch.sqrt(torch.var(z2, 0) + 1e-4)

        return (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2

    def _covariance(self, z1: Tensor, z2: Tensor) -> Tensor:

        z1 = z1 - torch.mean(z1, 0)
        z2 = z2 - torch.mean(z2, 0)

        cov_z1 = (z1.T @ z1) / (self.batch_size - 1)
        cov_z2 = (z2.T @ z2) / (self.batch_size - 1)

        mask = ~torch.eye(self.repr_size).bool()

        return (
            torch.sum(torch.pow(mask * cov_z1, 2))
            + torch.sum(torch.pow(mask * cov_z2, 2))
        ) / self.repr_size

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:

        inv = self._invariance(z1, z2)
        var = self._variance(z1, z2)
        cov = self._covariance(z1, z2)

        return self.vicreg_lambda * inv + self.vicreg_mu * var + self.vicreg_nu * cov


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def forward(x, y, batch_size, repr_size, sim_coeff, std_coeff, cov_coeff):
    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = 0
    cov_loss += off_diagonal(cov_x).pow_(2).sum().div(repr_size)
    cov_loss += off_diagonal(cov_y).pow_(2).sum().div(repr_size)

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def forward_byol(online_pred_one, online_pred_two, target_proj_two, target_proj_one):
    loss_one = loss_fn(online_pred_one, target_proj_two.detach())
    loss_two = loss_fn(online_pred_two, target_proj_one.detach())

    loss = loss_one + loss_two
    return loss.mean()


if __name__ == "__main__":
    l1 = VICRegLoss(16, 1024, 25, 25, 1)

    z1 = torch.randn(16, 1024)
    z2 = torch.randn(16, 1024)

    print(l1(z1, z2))
    print(forward(z1, z2, 16, 1024, 25, 25, 1))

    assert l1(z1, z2) == forward(z1, z2, 16, 1024, 25, 25, 1)

    q1 = torch.randn(16, 256)
    q2 = torch.randn(16, 256)

    z1 = torch.randn(16, 256)
    z2 = torch.randn(16, 256)

    l2 = BYOLLoss()

    print(l2(q1, q2, z1, z2))
    print(forward_byol(q1, q2, z1, z2))
