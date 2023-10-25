import json
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

import guidance
from joblib import Parallel, delayed
from lightning.fabric.utilities.seed import seed_everything
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.text2sql import text2sql


def prepare_schemas(
    dev: list[dict],
    resolution: Literal["database", "table", "column"],
    all_schemas: dict,
):
    for it in tqdm(dev):
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
        else:
            raise ValueError(f"Unknown resolution: {resolution}")

        it["schemas"] = schemas


def evaluate_text2sql(
    dataset: Literal["spider", "bird"] = "spider",
    resolution: Literal["database", "table", "column"] = "database",
    model_name: str = "gpt-3.5-turbo",
    override: bool = True,
):
    seed_everything(42)
    with Path(f"./data/{dataset}/schemas.json").open() as file:
        all_schemas = json.load(file)
    with Path(f"./data/{dataset}/test.json").open() as file:
        dev = json.load(file)

    pred_file = Path(f"./data/text2sql_results/{dataset}_{resolution}_{model_name}.txt")
    print(f"{dataset}_{resolution}_{model_name}")
    if override or not pred_file.exists():
        prepare_schemas(dev, resolution, all_schemas)
        results = Parallel(n_jobs=16)(
            delayed(text2sql)(it["question"], it["schemas"], model_name)
            for it in tqdm(dev)
        )
        preds, usages = zip(*results)

        with pred_file.open("w") as file:
            file.writelines(sql + "\n" for sql in preds)

        if usages[0] is not None:
            usage = sum(usages, Counter())
            print(usage)
            model = guidance.llms.OpenAI(model_name)
            print(model.get_usage_cost_usd(usage))

    ds_dir = f"data/{dataset}"
    if dataset == "spider":
        cmd = f"python src/vendor/test-suite-sql-eval/evaluation.py --pred {pred_file} --db {ds_dir}/databases --gold {ds_dir}/dev_gold.sql --table {ds_dir}/tables.json --etype exec"
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
        ...


if __name__ == "__main__":
    for dataset in [
        "spider",
        "bird",
        # "fiben",
    ]:
        for resolution in [
            "database",
            "table",
            "column",
        ]:
            evaluate_text2sql(dataset=dataset, resolution=resolution)
