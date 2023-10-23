from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer, get_scheduler

from src.models.modules import BarlowTwinsLoss, Pooler
from src.utils.collators import ContrastiveCollator


class SentenceEncoder(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int | None = None,
        pooler_type: Pooler.valid_types = "average",
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        scheduler_type: str = "linear",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.collate_fn = ContrastiveCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.pooler = Pooler(pooler_type=pooler_type)
        self.loss_func = BarlowTwinsLoss(dim=self.model.config.hidden_size)

    def forward(self, inputs) -> Any:
        outputs = self.model(**inputs)
        pooled_output = self.pooler(outputs, inputs.attention_mask)
        return pooled_output

    def common_step(self, batch) -> STEP_OUTPUT:
        x1, x2 = batch
        z1, z2 = self.forward(x1), self.forward(x2)
        loss = self.loss_func(z1, z2)
        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        loss = self.common_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> STEP_OUTPUT | None:
        loss = self.common_step(batch)
        self.log("validation/loss", loss, prog_bar=True)
        return loss

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT | None:
        loss = self.common_step(batch)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
