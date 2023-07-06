import json
import sys
from pathlib import Path
from typing import Literal

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.openai_with_usage import OpenAI
from src.utils.text2sql import text2sql

sys.path.append(str(Path(__file__).parent))

from prepare_data import get_databases_info


def gen_sqls(
    dataset: Literal["spider", "bird"] = "spider",
    schema: Literal["database", "table", "column"] = "database",
    model_name: str = "gpt-3.5-turbo",
):
    databases = get_databases_info(dataset)
    with open(f"./data/dev_{dataset}.json") as file:
        dev = json.load(file)

    # model = guidance.llms.OpenAI(model_name)
    model = OpenAI(model_name)
    # model.default_system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture."

    with open(f"./data/results/{dataset}_{schema}_{model_name}.txt", "w") as file:
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
            file.write(f"{sql}\n")

    if hasattr(model, "usage"):
        print(f"{dataset}_{schema}_{model_name}")
        print(model.usage)
        print(model.get_usage_cost_usd())


if __name__ == "__main__":
    for dataset in ["spider", "bird"]:
        for schema in ["database", "table", "column"]:
            gen_sqls(dataset=dataset, schema=schema)
