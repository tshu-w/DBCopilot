import torch
from torchmetrics import Metric


class F1Score(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[str]):
        for pred, target in zip(preds, target):
            pred = set(pred)
            target = set(target)
            self.tp += len(pred & target)
            self.fp += len(pred - target)
            self.fn += len(target - pred)

    def compute(self):
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return torch.tensor(0.0)
        else:
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
