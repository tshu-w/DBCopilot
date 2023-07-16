import torch
from torchmetrics import Metric


class Classification(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[str]):
        for pred, tgt in zip(preds, target):
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
            return self.tp / (self.tp + self.fp)


class Recall(Classification):
    def compute(self):
        denominator = self.tp + self.fn
        if denominator == 0:
            return torch.tensor(0.0)
        else:
            return self.tp / (self.tp + self.fn)


class F1Score(Classification):
    def compute(self):
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return torch.tensor(0.0)
        else:
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
