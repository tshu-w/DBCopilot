import json
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
from datasets import load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.utils.helpers import schema2graph


class Text2Schema(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "spider",
        train_data: Literal["original", "synthetic", "all"] = "synthetic",
        *,
        preprocessing_num_workers: int = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        train_files = []
        if train_data in ["original", "all"]:
            train_files.extend(
                list(map(str, sorted(Path("data").glob(f"{dataset}*/train.json"))))
            )
        if train_data in ["synthetic", "all"]:
            train_files.append(f"data/{dataset}/synthetic.json")

        self.data_files = {
            "train": train_files,
            **{
                "test"
                if f.parent.stem == dataset
                else f"test_{f.parent.stem[len(dataset) + 1:]}": [str(f)]
                for f in sorted(Path("data").glob(f"{dataset}*/test.json"))
            },
        }

    def prepare_data(self) -> None:
        # setup first to prevent datasets cache conflicts in multiple processes.
        self.setup()

    def setup(self, stage: str | None = None) -> None:
        if not hasattr(self, "datasets"):
            self.datasets = load_dataset("json", data_files=self.data_files)

            with Path(f"data/{self.dataset}/schemas.json").open("r") as f:
                self.schemas = json.load(f)

            if "validation" not in self.datasets:
                datasets_split = self.datasets["train"].train_test_split(
                    test_size=0.1, shuffle=True
                )
                self.datasets["train"] = datasets_split["train"]
                self.datasets["validation"] = datasets_split["test"]

            self.G = schema2graph(self.schemas)

            self.test_splits = [x for x in self.datasets.keys() if "test" in x]

            self.collate_fn = getattr(self.trainer.model, "collate_fn", None)
            self.collate_fn.G = self.G

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
            shuffle=False,
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
