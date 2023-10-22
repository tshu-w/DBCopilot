from typing import Literal

import torch
import torch.nn as nn


class Pooler(nn.Module):
    valid_types = Literal["cls", "average"]

    def __init__(
        self,
        pooler_type: valid_types,
    ):
        super().__init__()
        self.pooler_type = pooler_type.split("_")[0]

    def forward(self, outputs, attention_mask) -> torch.Tensor:
        if self.pooler_type == "cls":
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0]
        elif self.pooler_type == "average":
            last_hidden_state = outputs.last_hidden_state
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(
                dim=1
            ) / attention_mask.sum(dim=-1).unsqueeze(-1)

        return pooled_output
