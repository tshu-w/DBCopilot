import os
from functools import partial
from typing import Optional

import lightning.pytorch as pl
from datasets import load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess(batch, tokenizer, delimiters):
    initiator = delimiters["initiator"]
    separator = delimiters["separator"]
    terminator = delimiters["terminator"]

    targets = []
    for schema in batch["schema"]:
        for token in delimiters.values():
            assert token not in schema["database"]
            assert all(token not in t["name"] for t in schema["metadata"])
            assert all(
                token not in column
                for t in schema["metadata"]
                for column in t["columns"]
            )

        tables = separator.join(
            f"{initiator}{t['name']}{separator}{separator.join(t['columns'])}{terminator}"
            for t in schema["metadata"]
        )
        targets.append(
            f"{initiator}{schema['database']}{separator}{tables}{terminator}"
        )

    features = tokenizer(text=batch["question"], text_target=targets)

    for i, label in enumerate(features["labels"]):
        assert tokenizer.unk_token_id not in label

    return features


class Text2Schema(pl.LightningDataModule):
    def __init__(
        self,
        data_files: dict = {
            "train": ["data/train_spider.json"],
            "test": ["data/dev_spider.json"],
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
            if "validation" not in datasets:
                datasets_split = datasets["train"].train_test_split(
                    test_size=0.1, shuffle=True
                )
                datasets["train"] = datasets_split["train"]
                datasets["validation"] = datasets_split["test"]

            _preprocess = partial(
                preprocess,
                tokenizer=self.trainer.model.tokenizer,
                delimiters=self.trainer.model.hparams.delimiters,
            )
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

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
        )
