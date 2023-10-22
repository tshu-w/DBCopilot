import torch
import torch.nn as nn
import torch.nn.functional as F


# taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        dim: int,
        lambd: float = 0.005,
    ):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(dim, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class SelfContLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        z1, z2 = F.normalize(z1), F.normalize(z2)
        loss = (z1 * z2).sum(dim=-1).mean()
        return loss
