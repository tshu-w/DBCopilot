import asyncio
import json
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import guidance
from lightning.fabric.utilities.seed import seed_everything
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.text2sql import gather_with_concurrency, text2sql


def prepare_schemas(
    test: str,
    resolution: str,
    routing_file: str | None = None,
):
    with Path(f"./data/{test}/schemas.json").open() as file:
        all_schemas = json.load(file)
    with Path(f"./data/{test}/test.json").open() as file:
        dev = json.load(file)

    for idx, it in tqdm(list(enumerate(dev))):
        if resolution == "database":
            db = it["schema"]["database"]
            tbls = [
                {
                    "name": tbl["name"],
                    "columns": list(map(lambda c: c["name"], tbl["columns"])),
                }
                for tbl in all_schemas[db]
            ]
            schemas = [
                {
                    "name": db,
                    "tables": tbls,
                }
            ]
        elif resolution == "table":
            db = it["schema"]["database"]
            tbl_names = [t["name"] for t in it["schema"]["metadata"]]
            tbls = [
                {
                    "name": tbl["name"],
                    "columns": list(map(lambda c: c["name"], tbl["columns"])),
                }
                for tbl in all_schemas[db]
                if tbl["name"] in tbl_names
            ]
            schemas = [
                {
                    "name": db,
                    "tables": tbls,
                }
            ]
        elif resolution == "column":
            db = it["schema"]["database"]
            schemas = [
                {
                    "name": db,
                    "tables": it["schema"]["metadata"],
                }
            ]
        elif resolution == "random@5":
            schemas = []
            for db in [
                *random.choices(list(all_schemas.keys()), k=2),
                it["schema"]["database"],
                *random.choices(list(all_schemas.keys()), k=2),
            ]:
                tbls = [
                    {
                        "name": tbl["name"],
                        "columns": list(map(lambda c: c["name"], tbl["columns"])),
                    }
                    for tbl in all_schemas[db]
                ]
                schemas.append(
                    {
                        "name": db,
                        "tables": tbls,
                    }
                )
            assert len(schemas) == 5
        elif resolution.startswith("prediction"):
            with routing_file.open() as file:
                routing = json.load(file)

            schemas = []
            suffix = resolution.split("@")[1]
            if suffix == "cot":
                suffix = "5"

            gold_idx = -1
            for i in range(len(routing[idx]["pred_schemas"])):
                if (
                    routing[idx]["pred_schemas"][i]["database"]
                    == it["schema"]["database"]
                ):
                    gold_idx = i
                    break

            it["label"] = gold_idx
            if suffix != "-1":
                pred_schemas = routing[idx]["pred_schemas"][: int(suffix)]
            else:
                pred_schemas = [routing[idx]["pred_schemas"][gold_idx]]

            for pred_schema in pred_schemas:
                db = pred_schema["database"]
                tbls = pred_schema["tables"]
                schemas.append(
                    {
                        "name": db,
                        "tables": [
                            {
                                "name": tbl["name"],
                                "columns": list(
                                    map(lambda c: c["name"], tbl["columns"])
                                ),
                            }
                            for tbl in all_schemas[db]
                            if tbl["name"] in tbls
                        ],
                    }
                )
            assert len(schemas) <= 5
        elif resolution.startswith("baseline"):
            with routing_file.open() as file:
                routing = json.load(file)

            schemas = []
            pred_schemas = routing[idx]["pred_schemas"][:1]

            for pred_schema in pred_schemas:
                db = pred_schema["database"]
                tbls = pred_schema["tables"]
                schemas.append(
                    {
                        "name": db,
                        "tables": [
                            {
                                "name": tbl["name"],
                                "columns": list(
                                    map(lambda c: c["name"], tbl["columns"])
                                ),
                            }
                            for tbl in all_schemas[db]
                            if tbl["name"] in tbls
                        ],
                    }
                )
            assert len(schemas) <= 5
        else:
            raise ValueError(f"Unknown resolution: {resolution}")

        it["schemas"] = schemas

    return dev


def evaluate_text2sql(
    dataset: Literal["spider", "bird"],
    test: str,
    resolution: str,
    routing_file: Path | None = None,
    model_name: str = "gpt-3.5-turbo-16k",
    override: bool = True,
):
    seed_everything(42)
    pred_file = Path(f"./data/text2sql_results/{test}_{resolution}_{model_name}.txt")
    print(f"{test}_{resolution}_{model_name}")
    if override or not pred_file.exists():
        model = guidance.llms.OpenAI(model_name)
        dev = prepare_schemas(test, resolution, routing_file)
        cot = True if resolution.endswith("cot") else False
        preds = asyncio.run(
            gather_with_concurrency(
                32, *[text2sql(it["question"], it["schemas"], model, cot) for it in dev]
            )
        )
        if resolution.endswith("@cot"):
            preds, pred_idxes = list(zip(*preds))
            labels = [it["label"] for it in dev]
            print("Accuracy: ", accuracy_score(labels, pred_idxes))

        with pred_file.open("w") as file:
            file.writelines(sql + "\n" for sql in preds)

        print(model.usage)
        print(model.get_usage_cost_usd())

    ds_dir = f"data/{dataset}"
    ts_dir = f"data/{test}"
    if dataset == "spider":
        cmd = f"python src/vendor/test-suite-sql-eval/evaluation.py --pred {pred_file} --db {ds_dir}/databases --gold {ts_dir}/dev_gold.sql --table {ds_dir}/tables.json --etype exec"
        result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)
        print(result.stdout.decode("utf-8"))

    elif dataset == "bird":
        # https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L166
        with tempfile.TemporaryDirectory() as tmpdir:
            preds = {}
            with pred_file.open() as file:
                for i, ln in enumerate(file):
                    preds[i] = f"{ln}\t----- bird -----\t{dev[i]['schema']['database']}"

            dest_file = Path(tmpdir) / "predict_dev.json"
            with dest_file.open("w") as f:
                json.dump(preds, f, indent=4)

            cmd = f"python src/vendor/DAMO-ConvAI/bird/llm/src/evaluation.py --predicted_sql_path {tmpdir}/ --ground_truth_path {ds_dir}/ --db_root_path {ds_dir}/databases/ --diff_json_path {ds_dir}/dev_gold.json --num_cpus 8 --data_mode dev"
            result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)
            print(result.stdout.decode("utf-8"))
    elif dataset == "fiben":
        raise NotImplementedError


if __name__ == "__main__":
    datasets = {
        "spider": ["spider", "spider_syn", "spider_realistic", "spider_dr"],
        "bird": ["bird"],
        # "fiben": ["fiben"],
    }
    routing_dirs = {
        "spider": Path("./results/test/silver-sun-65/k5ou2ykv"),
        "bird": Path("./results/test/happy-dew-65/ey4tt3n2"),
        # "fiben": Path("./results/test/electric-donkey-66/sj2wkc2b"),
    }
    for dataset, tests in datasets.items():
        for test in tests:
            for resolution in [
                "database",
                "table",
                "column",
                "random@5",
                "prediction@1",
                "prediction@5",
                "prediction@-1",
                "prediction@cot",
                "baseline@crush4sql_bm25",
                "baseline@dpr",
            ]:
                if resolution.startswith("prediction"):
                    routing_file = (
                        routing_dirs[dataset] / f'{test.replace(dataset, "test")}.json'
                    )
                elif resolution.startswith("baseline"):
                    if resolution.endswith("crush4sql_bm25"):
                        routing_file = (
                            Path("results")
                            / "retrieval"
                            / f"crush4sql_{test}_sparse_false.json"
                        )
                    elif resolution.endswith("dpr"):
                        routing_file = (
                            Path("results") / "retrieval" / f"{test}_dense_true.json"
                        )
                else:
                    routing_file = None

                evaluate_text2sql(
                    dataset=dataset,
                    test=test,
                    resolution=resolution,
                    routing_file=routing_file,
                )
