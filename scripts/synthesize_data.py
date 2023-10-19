import json
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from datasets import Dataset
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from walker import random_walks

sys.path.append(str(Path(__file__).parents[1]))

from src.models import SchemaQuestioning
from src.utils.helpers import schema2graph, snode


def synthesize_data(
    dataset: str,
    ckpt_path: str,
    k: int = 20,
    p: float = 1.0,
    q: float = 1.0,
):
    seed_everything(42)
    with Path(f"./data/{dataset}/schemas.json").open() as f:
        schemas = json.load(f)
    G = schema2graph(schemas)

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
    max_tbls = 0
    for it in data:
        counters["table"][len(it["schema"]["metadata"])] += 1
        max_tbls = max(max_tbls, len(it["schema"]["metadata"]))
        for table in it["schema"]["metadata"]:
            counters["column"][len(table["columns"])] += 1

    node2idx = {n: i for i, n in enumerate(G)}
    idx2node = {i: n for i, n in enumerate(G)}
    X = random_walks(
        G,
        n_walks=k * len(G),
        walk_len=2 + 2 * max_tbls,
        start_nodes=[node2idx[snode]],
        p=p,
        q=q,
        verbose=False,
    )
    walks = [[idx2node[idx] for idx in x] for x in X]

    sythetic_data = []
    for walk in walks:
        db, *tbls = map(lambda x: x.name, OrderedDict.fromkeys(walk[1:]))
        schema = {"database": db, "metadata": []}

        tbl_nums = random.choices(
            list(counters["table"]), weights=counters["table"].values()
        )
        tbls = tbls[: min(tbl_nums[0] + 1, len(tbls))]
        for tbl in tbls:
            for table in schemas[db]:
                if tbl == table["name"]:
                    # col_nums = random.choices(
                    #     list(counters["column"]), weights=counters["column"].values()
                    # )
                    # col_num = min(col_nums[0], len(table["columns"]))
                    # columns = random.sample(table["columns"], k=col_num)
                    # columns = list(map(lambda x: x["name"], columns))
                    columns = list(map(lambda x: x["name"], table["columns"]))
                    schema["metadata"].append(
                        {
                            "name": tbl,
                            "columns": columns,
                        }
                    )

        synthetic_data.append({"schema": schema, "sql": ""})

    ds = Dataset.from_list(sythetic_data)
    model = SchemaQuestioning.load_from_checkpoint(ckpt_path)
    dataloader = DataLoader(
        ds, batch_size=256, collate_fn=model.collate_fn, num_workers=32
    )
    trainer = Trainer(logger=False)
    preditions = trainer.predict(model, dataloader)
    preditions = [t for lst in preditions for t in lst]
    for question, generated in zip(preditions, sythetic_data):
        generated["question"] = question

    with Path(f"./data/{dataset}/sythetic.json").open("w") as f:
        json.dump(sythetic_data, f, indent=2)


if __name__ == "__main__":
    datasets = [
        "spider",
        "bird",
        "wikisql",
        "fiben",
    ]
    ckpt_paths = [
        "results/fit/classic-lake-5/fk0oxh78/checkpoints/epoch=29-step=28290.ckpt"
    ]

    for dataset, ckpt_path in zip(datasets, ckpt_paths * len(datasets)):
        synthesize_data(dataset, ckpt_path, k=40)
