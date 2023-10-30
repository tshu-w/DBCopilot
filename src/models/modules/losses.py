from typing import Literal

import torch
import torch.nn as nn
from pytorch_metric_learning import losses


class NTXentLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        direction: Literal["single", "both"] = "both",
    ):
        super().__init__()
        self.loss_func = losses.NTXentLoss(temperature)
        self.direction = direction

    def forward(self, z1, z2):
        labels = torch.arange(len(z1), device=z1.device)
        if self.direction == "single":
            return self.loss_func(
                embeddings=z1, labels=labels, ref_emb=z2, ref_labels=labels.clone()
            )
        else:
            embeddings = torch.cat([z1, z2])
            labels = torch.cat([labels, labels])
            return self.loss_func(embeddings, labels)


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
