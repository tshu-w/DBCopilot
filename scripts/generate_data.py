import json
import random
import sys
from collections import Counter
from functools import partial
from pathlib import Path

from datasets import Dataset
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parents[1]))

from src.datamodules.schema2text import preprocess
from src.models import Schema2Query

sys.path.append(str(Path(__file__).parent))

from prepare_data import get_dataset_schemas


def generate_data(dataset: str, ckpt_path: str, k: int = 200):
    seed_everything(42)
    databases = get_dataset_schemas(dataset)
    files = list(Path("./data").glob(f"{dataset}*/*.json"))
    files = [f for f in files if not str(f).endswith("schemas.json")]
    data = []
    for f in files:
        with f.open() as f:
            data.extend(json.load(f))

    counters = {
        "table": Counter(),
        "column": Counter(),
    }
    for it in data:
        db = it["schema"]["database"]
        databases[db]

        counters["table"][len(it["schema"]["metadata"])] += 1
        for table in it["schema"]["metadata"]:
            counters["column"][len(table["columns"])] += 1

    pseudo_data = []
    for db, tables in databases.items():
        tbl_nums = random.choices(
            list(counters["table"]), weights=counters["table"].values(), k=k
        )
        for tbl_num in tbl_nums:
            schema = {"database": db, "metadata": []}
            sampled_tbls = random.sample(tables, k=min(tbl_num, len(tables)))
            for tbl in sampled_tbls:
                col_num = random.choices(
                    list(counters["column"]), weights=counters["column"].values()
                )[0]
                schema["metadata"].append(
                    {
                        "name": tbl["name"],
                        "columns": random.sample(
                            tbl["columns"], k=min(col_num, len(tbl["columns"]))
                        ),
                    }
                )

            pseudo_data.append({"schema": schema, "query": ""})

    ds = Dataset.from_list(pseudo_data)
    model = Schema2Query.load_from_checkpoint(ckpt_path)
    _preprocess = partial(
        preprocess,
        tokenizer=model.tokenizer,
        max_length=model.hparams.max_length,
    )
    ds = ds.map(
        _preprocess,
        batched=True,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    dataloader = DataLoader(ds, batch_size=64, collate_fn=model.collate_fn)
    trainer = Trainer(logger=False)
    preditions = trainer.predict(model, dataloader)
    preditions = [t for lst in preditions for t in lst]
    for question, generated in zip(preditions, pseudo_data):
        generated["question"] = question

    with Path(f"./data/{dataset}/pseudo_{k}.json").open("w") as f:
        json.dump(pseudo_data, f, indent=2)


if __name__ == "__main__":
    for dataset, ckpt_path in zip(
        ["spider", "bird"],
        [
            "results/fit/northern-aardvark-102/duf6v9ki/checkpoints/epoch=49-step=22050.ckpt",
            "results/fit/icy-pond-103/6lkridkl/checkpoints/epoch=49-step=13300.ckpt",
        ],
    ):
        for k in [100, 200, 400]:
            generate_data(dataset, ckpt_path, k=k)
