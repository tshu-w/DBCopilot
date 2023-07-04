import os
from functools import partial
from typing import Optional

import lightning.pytorch as pl
from datasets import load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess(batch, tokenizer):
    targets = []
    for db, metadata in zip(batch["database"], batch["metadata"]):
        tables = " ".join(f"({t['name']} {' '.join(t['columns'])})" for t in metadata)
        targets.append(f"({db} {tables})")

    features = tokenizer(text=batch["question"], text_target=targets)

    return features


class Text2Schema(pl.LightningDataModule):
    def __init__(
        self,
        data_files: dict = {
            "train": ["data/train_spider.json"],
            "validation": ["data/dev_spider.json"],
        },
        *,
        preprocessing_num_workers: int = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_files = data_files

    def prepare_data(self) -> None:
        # setup first to prevent datasets cache conflicts in multiple processes.
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            datasets = load_dataset("json", data_files=self.data_files)
            self.trainer.model.tokenizer.deprecation_warnings[
                "Asking-to-pad-a-fast-tokenizer"
            ] = True
            _preprocess = partial(preprocess, tokenizer=self.trainer.model.tokenizer)
            self.datasets = datasets.map(
                _preprocess,
                batched=True,
                remove_columns=datasets["train"].column_names,
                num_proc=self.hparams.preprocessing_num_workers,
            )

        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["validation"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
        )
