import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.utils.helpers import schema2label

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess(batch, tokenizer, max_length, add_db):
    inputs, targets = [], []
    for question, schema in zip(batch["question"], batch["schema"]):
        target = schema2label(schema, separator=tokenizer.sep_token, add_db=add_db)
        targets.append(target)

        if question is None:
            inputs.append("")
        else:
            inputs.append(question)

    features = tokenizer(
        text=inputs,
        text_target=targets,
        max_length=max_length,
        truncation=True,
    )

    for i, label in enumerate(features["labels"]):
        assert tokenizer.unk_token_id not in label

    return features


class Text2Schema(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "spider",
        pseudo: Union[bool, str] = True,
        add_db: bool = True,
        *,
        preprocessing_num_workers: int = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.data_files = {
            "train": list(
                map(str, sorted(Path("data").glob(f"{dataset}*/train.json")))
            ),
            **{
                "test"
                if f.parent.stem == dataset
                else f"test_{f.parent.stem[len(dataset):]}": [str(f)]
                for f in sorted(Path("data").glob(f"{dataset}*/test.json"))
            },
        }
        if pseudo:
            pseudo = "pseudo_200" if pseudo is True else pseudo
            self.data_files["train"].append(f"data/{dataset}/{pseudo}.json")

    def prepare_data(self) -> None:
        # setup first to prevent datasets cache conflicts in multiple processes.
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            datasets = load_dataset("json", data_files=self.data_files)

            with Path(f"data/{self.dataset}/schemas.json").open("r") as f:
                self.schemas = json.load(f)

            if self.hparams.add_db is False:
                self.tbl2db = defaultdict(list)
                for database, tables in self.schemas.items():
                    for table in tables:
                        self.tbl2db[table["name"]].append(database)
            else:
                self.tbl2db = {}

            list_of_schemas = [
                {"schema": {"database": k, "metadata": v}}
                for k, v in self.schemas.items()
            ]
            schemas_ds = Dataset.from_pandas(pd.DataFrame(list_of_schemas))
            datasets["train"] = concatenate_datasets([datasets["train"], schemas_ds])

            if "validation" not in datasets:
                datasets_split = datasets["train"].train_test_split(
                    test_size=0.1, shuffle=True
                )
                datasets["train"] = datasets_split["train"]
                datasets["validation"] = datasets_split["test"]

            _preprocess = partial(
                preprocess,
                tokenizer=self.trainer.model.tokenizer,
                max_length=self.trainer.model.hparams.max_length,
                add_db=self.hparams.add_db,
            )
            self.datasets = datasets.map(
                _preprocess,
                batched=True,
                remove_columns=datasets["train"].column_names,
                num_proc=self.hparams.preprocessing_num_workers,
                load_from_cache_file=False,
            )

            self.test_splits = [x for x in self.datasets.keys() if "test" in x]

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
        test_dataloaders = [
            DataLoader(
                dataset=self.datasets[x],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
                persistent_workers=self.hparams.num_workers > 0,
                shuffle=False,
            )
            for x in self.test_splits
        ]

        return test_dataloaders[0] if len(test_dataloaders) == 1 else test_dataloaders
