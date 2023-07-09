import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.openai_with_usage import OpenAI
from src.utils.text2sql import text2sql

sys.path.append(str(Path(__file__).parent))

from prepare_data import get_databases_info


def evaluate_text2sql(
    dataset: Literal["spider", "bird"] = "spider",
    schema: Literal["database", "table", "column"] = "database",
    model_name: str = "gpt-3.5-turbo",
    override: bool = True,
):
    databases = get_databases_info(dataset)
    with Path(f"./data/dev_{dataset}.json").open() as file:
        dev = json.load(file)

    # model = guidance.llms.OpenAI(model_name)
    model = OpenAI(model_name)
    # model.default_system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture."

    pred_file = Path(f"./data/results/{dataset}_{schema}_{model_name}.txt")
    print(f"{dataset}_{schema}_{model_name}")
    if override or not pred_file.exists():
        preds = []
        for it in tqdm(dev):
            if schema == "database":
                database = databases[it["schema"]["database"]]
            elif schema == "table":
                table_names = [t["name"] for t in it["schema"]["metadata"]]
                database = [
                    t
                    for t in databases[it["schema"]["database"]]
                    if t["name"] in table_names
                ]
            elif schema == "column":
                database = it["schema"]["metadata"]
            else:
                raise ValueError("schema must be one of 'database', 'table', 'column'")

            sql = text2sql(
                question=it["question"],
                database=database,
                model=model,
            )
            preds.append(sql)

        with pred_file.open("w") as file:
            file.writelines(sql + "\n" for sql in preds)

        if hasattr(model, "usage"):
            print(model.usage)
            print(model.get_usage_cost_usd())

    if dataset == "spider":
        root_dir = "data/spider"
        cmd = f"python src/vendor/test-suite-sql-eval/evaluation.py --pred {pred_file} --db {root_dir}/database --gold {root_dir}/dev_gold.sql --table {root_dir}/tables.json --etype exec"
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

            root_dir = "data/bird/dev"
            cmd = f"python src/vendor/DAMO-ConvAI/bird/llm/src/evaluation.py --predicted_sql_path {tmpdir}/ --ground_truth_path {root_dir}/ --db_root_path {root_dir}/dev_databases/ --diff_json_path {root_dir}/dev.json --num_cpus 8 --data_mode dev"
            result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)
            print(result.stdout.decode("utf-8"))


if __name__ == "__main__":
    for dataset in ["spider", "bird"]:
        for schema in ["database", "table", "column"]:
            evaluate_text2sql(dataset=dataset, schema=schema)
