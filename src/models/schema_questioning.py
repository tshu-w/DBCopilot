import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from peft import get_peft_config, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
)

from src.utils.collators import Schema2TextCollator


class SchemaQuestioning(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        peft_config: dict | None = None,
        generator_config: dict | None = None,
        max_length: int | None = 512,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-4,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            self.mode = "seq2seq"
        except ValueError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.mode = "causal"

        if peft_config is not None:
            peft_config = get_peft_config(peft_config)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        self.generator_config = generator_config or {"max_new_tokens": 512}

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.collate_fn = Schema2TextCollator(
            tokenizer=self.tokenizer,
            model_mode=self.mode,
            max_length=max_length,
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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model.generate(**batch, **self.hparams.generator_config)
        if self.mode == "causal":
            outputs = [
                opt[len(ipt) :]
                for ipt, opt in zip(batch["input_ids"], outputs, strict=True)
            ]

        pred_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return pred_texts

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.trainer.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.trainer.model.named_parameters()
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
