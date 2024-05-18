import torch
from torchmetrics import Metric


class Classification(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        top_k: int | None = None,
    ):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.top_k = top_k

    def update(self, preds: list[list[str]], target: list[list[str]]):
        for pred, tgt in zip(preds, target, strict=True):
            if self.top_k is not None:
                pred = pred[: self.top_k]
            pred = set(pred)
            tgt = set(tgt)
            self.tp += len(pred & tgt)
            self.fp += len(pred - tgt)
            self.fn += len(tgt - pred)


class Precision(Classification):
    def compute(self):
        denominator = self.tp + self.fp
        if denominator == 0:
            return torch.tensor(0.0)
        else:
            return self.tp / (self.tp + self.fp) * 100


class Recall(Classification):
    def compute(self):
        denominator = self.tp + self.fn
        if denominator == 0:
            return torch.tensor(0.0)
        else:
            return self.tp / (self.tp + self.fn) * 100


class F1Score(Classification):
    def compute(self):
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return torch.tensor(0.0)
        else:
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn) * 100
