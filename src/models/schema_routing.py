import json
from collections import Counter, OrderedDict, defaultdict
from functools import partial
from itertools import chain
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

from src.models.modules import ConstraintDecoder, Recall
from src.utils.collators import Text2SchemaCollator
from src.utils.helpers import chunks, deserialize_schema


def prefix_allowed_tokens_fn(_batch_id, sent, constraint_decoder):
    allowed_tokens = constraint_decoder(sent.tolist())
    return allowed_tokens


class SchemaRouting(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        generator_config: dict = {
            "max_new_tokens": 512,
            "constraint_decoding": True,
        },
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

        db_metrics = MetricCollection(
            {"dR@1": Recall(top_k=1), "dR@5": Recall(top_k=5)}
        )
        tbl_metrics = MetricCollection(
            {"tR@5": Recall(top_k=5), "tR@25": Recall(top_k=25)}
        )
        self.metrics = torch.nn.ModuleDict()
        for step in ["validation", "test"]:
            self.metrics[step] = torch.nn.ModuleDict(
                {
                    "db": db_metrics.clone(prefix=f"{step}/"),
                    "tbl": tbl_metrics.clone(prefix=f"{step}/"),
                }
            )

        self.pre_dataloader_idx = -1

        self.outputs = defaultdict(list)

    def setup(self, stage: str) -> None:
        # prepare prefix_allowed_tokens_fn for constraint decoding
        constraint_decoding = self.generator_config.pop("constraint_decoding", False)
        if (
            constraint_decoding
            and self.generator_config.get("prefix_allowed_tokens_fn", None) is None
        ):
            constraint_decoder = ConstraintDecoder(
                tokenizer=self.tokenizer,
                G=self.trainer.datamodule.G,
            )
            partial_prefix_allowed_tokens_fn = partial(
                prefix_allowed_tokens_fn, constraint_decoder=constraint_decoder
            )
            self.generator_config[
                "prefix_allowed_tokens_fn"
            ] = partial_prefix_allowed_tokens_fn

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
        self, batch, step: str, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT | None:
        # Reset metrics between different test dataloaders
        if (
            step == "test"
            and (prefix := f"{self.trainer.datamodule.test_splits[dataloader_idx]}/")
            != self.metrics[step]["db"].prefix
        ):
            self.metrics[step]["db"].reset()
            self.metrics[step]["tbl"].reset()
            self.metrics[step]["db"].prefix = prefix
            self.metrics[step]["tbl"].prefix = prefix
        else:
            prefix = self.metrics[step]["db"].prefix

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
        pred_texts = list(chunks(pred_texts, chunk_size))
        pred_schemas = list(chunks(pred_schemas, chunk_size))

        global_bs = self.trainer.datamodule.hparams.batch_size
        raw_data = self.trainer.datamodule.datasets[prefix[:-1]][
            batch_idx * global_bs : (batch_idx + 1) * global_bs
        ]
        raw_data["pred_texts"] = pred_texts
        raw_data["pred_schemas"] = pred_schemas
        preds = [{k: raw_data[k][i] for k in raw_data} for i in range(batch_size)]
        self.outputs[prefix[:-1]].extend(preds)

        self.update_metrics(preds, metric_key=step)
        self.log_dict(self.metrics[step]["db"], prog_bar=True, add_dataloader_idx=False)
        self.log_dict(
            self.metrics[step]["tbl"], prog_bar=True, add_dataloader_idx=False
        )

        loss = self.common_step(batch)
        self.log(f"{prefix}loss", loss, prog_bar=True, add_dataloader_idx=False)

        return loss

    def validation_step(self, batch, batch_idx: int) -> STEP_OUTPUT | None:
        return self.evaluation_step(batch, "validation", batch_idx)

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT | None:
        return self.evaluation_step(batch, "test", batch_idx, dataloader_idx)

    def update_metrics(self, preds: list[dict], metric_key: str):
        pred_databases = []
        pred_tables = []
        for it in preds:
            databases = list(
                OrderedDict.fromkeys(s["database"] for s in it["pred_schemas"])
            )
            tables = [
                f"{db}.{t}"
                for db in databases
                for t, _ in Counter(
                    chain(
                        *(
                            s["tables"]
                            for s in it["pred_schemas"]
                            if s["database"] == db
                        )
                    )
                ).most_common()
            ]
            pred_databases.append(databases)
            pred_tables.append(tables)

        tgt_databases = []
        tgt_tables = []
        for it in preds:
            databases = [it["schema"]["database"]]
            tables = [
                f'{it["schema"]["database"]}.{t["name"]}'
                for t in it["schema"]["metadata"]
            ]
            tgt_databases.append(databases)
            tgt_tables.append(tables)

        self.metrics[metric_key]["db"](pred_databases, tgt_databases)
        self.metrics[metric_key]["tbl"](pred_tables, tgt_tables)

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
