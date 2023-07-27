from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)


class Schema2Query(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        generator_config: dict = {
            "max_new_tokens": 512,
        },
        delimiters: dict[str, str] = {
            "initiator": "<(>",
            "separator": "< >",
            "terminator": "<)>",
        },
        *,
        max_length: int = 512,
        weight_decay: float = 0.0,
        learning_rate: float = 2e-5,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, verbose=False
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.generator_config = generator_config

        num_added = self.tokenizer.add_tokens(
            list(delimiters.values()), special_tokens=True
        )
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.collate_fn = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def common_step(self, batch):
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        loss = self.common_step(batch)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss = self.common_step(batch)
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        loss = self.common_step(batch)
        self.log("test/loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model.generate(**batch, **self.hparams.generator_config)
        pred_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pred_texts

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
