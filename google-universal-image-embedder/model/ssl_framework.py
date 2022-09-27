from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
import pytorch_lightning as pl


class SimCLRLoss(nn.Module):
    def __init__(self, batch_size: int, temperature: float):

        super().__init__()

        self.batch_size = batch_size
        self.temperature = temperature

    def _cosine_similarity_matrix(self, z: Tensor) -> Tensor:

        return F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2])
        exp = torch.exp(self._cosine_similarity_matrix(z) / self.temperature)

        mask = torch.eye(2 * self.batch_size).bool()
        pmask = mask.roll(self.batch_size, 0)
        nmask = ~mask

        return torch.mean(
            -torch.log(torch.sum(pmask * exp, 0) / torch.sum(nmask * exp, 1))
        )


class ContrastiveLoss(nn.Module):
    """
    Using this for benchmark (DELETE LATER)
    https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float(),
        )

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature
        )

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


if __name__ == "__main__":
    from time import time_ns

    l1 = SimCLRLoss(16, 0.5)
    l2 = ContrastiveLoss(16, 0.5)

    z1 = torch.randn(16, 1024)
    z2 = torch.randn(16, 1024)

    assert l1(z1, z2) == l2(z1, z2)
