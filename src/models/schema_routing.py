import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
)

from src.models.modules import Recall
from src.utils.collators import Text2SchemaCollator
from src.utils.helpers import chunks, deserialize_schema


class SchemaRouting(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        generator_config: dict = {
            "max_new_tokens": 512,
        },
        *,
        sep_token: str = "<sep>",
        max_length: int | None = 512,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-4,
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

        if self.tokenizer.sep_token is None:
            self.tokenizer.add_special_tokens({"sep_token": sep_token})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.collate_fn = Text2SchemaCollator(
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        metrics = MetricCollection({"dR": Recall(), "tR": Recall()})
        self.metrics = torch.nn.ModuleDict()
        for step in ["val", "test"]:
            self.metrics[step] = metrics.clone(prefix=f"{step}/")

        self.pre_dataloader_idx = -1

        self.outputs = defaultdict(list)

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

    def evaluation_step(
        self, batch, step: str, dataloader_idx: int = 0
    ) -> STEP_OUTPUT | None:
        # Reset metrics between different test dataloaders
        if (
            step == "test"
            and (prefix := f"{self.trainer.datamodule.test_splits[dataloader_idx]}/")
            != self.metrics[step].prefix
        ):
            self.metrics[step].reset()
            self.metrics[step].prefix = prefix
        else:
            prefix = self.metrics[step].prefix

        outputs = self.model.generate(**batch, **self.generator_config)
        pred_texts = [
            self.postprocess_text(s)
            for s in self.tokenizer.batch_decode(
                outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        ]
        _label2schema = partial(
            deserialize_schema,
            separator=self.tokenizer.sep_token,
        )
        pred_schemas = [_label2schema(s) for s in pred_texts]
        batch_size = len(batch["input_ids"])
        chunk_size = len(pred_schemas) // batch_size
        self.outputs[f"{prefix[:-1]}_pred_texts"].extend(
            chunks(pred_texts, chunk_size) if chunk_size > 1 else pred_texts
        )
        self.outputs[f"{prefix[:-1]}_pred_schemas"].extend(
            chunks(pred_schemas, chunk_size) if chunk_size > 1 else pred_schemas
        )

        labels = torch.where(
            batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id
        )
        target_texts = [
            self.postprocess_text(s)
            for s in self.tokenizer.batch_decode(
                labels, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        ]
        target_schemas = [_label2schema(s) for s in target_texts]
        self.outputs[f"{prefix[:-1]}_tgt_texts"].extend(target_texts)
        self.outputs[f"{prefix[:-1]}_tgt_schemas"].extend(target_schemas)

        self.update_metrics(pred_schemas, target_schemas, metric_key=step)
        self.log_dict(self.metrics[step], prog_bar=True, add_dataloader_idx=False)

        loss = self.common_step(batch)
        self.log(f"{step}/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> STEP_OUTPUT | None:
        return self.evaluation_step(batch, step="val")

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT | None:
        return self.evaluation_step(batch, step="test", dataloader_idx=dataloader_idx)

    def update_metrics(
        self, pred_schemas: list[dict], target_schemas: list[dict], metric_key: str
    ):
        def merge_nested_list(
            nested_lst: list[list[str]], step: int
        ) -> list[list[str]]:
            merged = []
            for i in range(0, len(nested_lst), step):
                merged.append([it for lst in nested_lst[i : i + step] for it in lst])
            return merged

        merge_step = len(pred_schemas) // len(target_schemas)

        pred_databases = [[d for d in s] for s in pred_schemas]
        target_databases = [[d for d in s] for s in target_schemas]
        pred_databases = merge_nested_list(pred_databases, step=merge_step)
        self.metrics[metric_key]["dR"](pred_databases, target_databases)

        pred_tables = [[f"{d}.{t}" for d in s for t in s[d]] for s in pred_schemas]
        target_tables = [[f"{d}.{t}" for d in s for t in s[d]] for s in target_schemas]
        pred_tables = merge_nested_list(pred_tables, step=merge_step)
        self.metrics[metric_key]["tR"](pred_tables, target_tables)

    def postprocess_text(self, s: str) -> str:
        for token in ["bos_token", "pad_token", "eos_token"]:
            if getattr(self.tokenizer, token) is not None:
                s = s.replace(getattr(self.tokenizer, token), "")

        return s.strip()

    def on_validation_epoch_end(self) -> None:
        for k, v in self.outputs.items():
            pth = Path(self.trainer.log_dir) / f"{k}.json"
            with pth.open("w") as f:
                json.dump(v, f, indent=2)

        self.outputs.clear()

    def on_test_epoch_end(self) -> None:
        for k, v in self.outputs.items():
            pth = Path(self.trainer.log_dir) / f"{k}.json"
            with pth.open("w") as f:
                json.dump(v, f, indent=2)

        self.outputs.clear()

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
