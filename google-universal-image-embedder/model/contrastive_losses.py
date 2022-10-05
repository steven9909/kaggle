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

    def l(self, q: Tensor, z: Tensor) -> Tensor:

        return 2 - 2 * torch.sum(q * z, dim=1) / (
            torch.norm(q, p=2, dim=1) * torch.norm(z, p=2, dim=1)
        )

    def forward(self, q1: Tensor, q2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
        """
        q1: batch of N augumented (t), predicted embeddings (online)
        q2: batch of N augumented (t') predicted embeddings (online)
        z1: batch of N augumented (t') projected embeddings (target)
        z2: batch of N augumented (t) projected embeddings (target)
        """

        return torch.mean(self.l(q1, z1) + self.l(q2, z2))


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

if __name__ == "__main__":
