import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from src.utils.text2sql import text2sql


def spider(model_name="gpt-3.5-turbo-0613", dir=Path("./data/spider")):
    dev = pd.read_json(dir / "dev.json", orient="records").to_dict(orient="records")
    raw_tables = pd.read_json(dir / "tables.json", orient="records")

    databases = {}
    for db in raw_tables.itertuples():
        tables = []
        for i, table_name in enumerate(db.table_names_original):
            columns = set(
                map(
                    lambda x: x[1],
                    filter(lambda x: x[0] == i, db.column_names_original),
                )
            )
            tables.append({"name": table_name, "columns": columns})

        databases[db.db_id] = tables

    with (dir / f"{model_name}.txt").open(mode="w") as file:
        for instance in tqdm(dev):
            sql = text2sql(
                question=instance["question"],
                database=databases[instance["db_id"]],
                model_name=model_name,
            )
            file.write(f"{sql}\n")


if __name__ == "__main__":
    spider()
