import json
import sys
from pathlib import Path
from typing import Literal

from sqlglot import exp, parse_one
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))


def extract_metadata(sql_query: str, database: list[dict]):
    parsed_query = parse_one(sql_query, read="sqlite")

    tables = [t.name.lower() for t in parsed_query.find_all(exp.Table)]
    columns = [c.name.lower() for c in parsed_query.find_all(exp.Column)]

    metadata = [
        {
            "name": t["name"],
            "columns": [c for c in t["columns"] if c in columns],
        }
        for t in database
        if t["name"] in tables
    ]

    return metadata


def get_databases_info(
    dataset: Literal["spider", "bird"] = "spider",
) -> dict:
    databases = {}
    if dataset == "spider":
        path = Path("./data/raw/spider")
        with (path / "tables.json").open(mode="r") as file:
            raw_tables = json.load(file)

        for db in raw_tables:
            tables = []
            for i, table_name in enumerate(db["table_names_original"]):
                columns = list(
                    map(
                        lambda x: x[1].lower(),
                        filter(lambda x: x[0] == i, db["column_names_original"]),
                    )
                )
                tables.append({"name": table_name.lower(), "columns": columns})

            databases[db["db_id"]] = tables
    elif dataset == "bird":
        path = Path("./data/raw/bird")
        with (path / "train" / "train_tables.json").open() as f:
            raw_tables = json.load(f)

        with (path / "dev" / "dev_tables.json").open() as f:
            raw_tables.extend(json.load(f))

        databases = {}
        for db in raw_tables:
            tables = []
            for i, table_name in enumerate(db["table_names_original"]):
                columns = list(
                    map(
                        lambda x: x[1].lower(),
                        filter(lambda x: x[0] == i, db["column_names_original"]),
                    )
                )
                tables.append({"name": table_name.lower(), "columns": columns})

            databases[db["db_id"]] = tables
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return databases


def spider(path=Path("./data/raw/spider/")):
    databases = get_databases_info("spider")

    train_files = list(path.glob("train_*.json"))
    train_data = []
    for f in train_files:
        with f.open() as f:
            train_data.extend(json.load(f))

    for record in tqdm(train_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["schema"] = {
            "database": record.pop("db_id"),
            "metadata": metadata,
        }
        for key in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
            record.pop(key, None)

    with (path.parent / "spider_train.json").open("w") as f:
        json.dump(train_data, f, indent=2)

    with (path / "dev.json").open() as f:
        dev_data = json.load(f)

    for record in tqdm(dev_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["schema"] = {
            "database": record.pop("db_id"),
            "metadata": metadata,
        }
        for key in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
            record.pop(key, None)

    with (path.parent / "spider_test.json").open("w") as f:
        json.dump(dev_data, f, indent=2)

    with (path.parent / "spider_schemas.json").open("w") as f:
        json.dump(databases, f, indent=2)


def spider_variants():
    databases = get_databases_info("spider")

    with Path("./data/raw/spider-syn/train_spider.json").open() as f:
        train_data = json.load(f)

    for record in tqdm(train_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["schema"] = {
            "database": record.pop("db_id"),
            "metadata": metadata,
        }
        record["question"] = record.pop("SpiderSynQuestion")
        for key in ["SpiderQuestion"]:
            record.pop(key, None)

    with Path("./data/raw/spider_train_syn.json").open("w") as f:
        json.dump(train_data, f, indent=2)

    with Path("./data/raw/spider-syn/dev.json").open() as f:
        dev_data = json.load(f)

    for record in tqdm(dev_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["schema"] = {
            "database": record.pop("db_id"),
            "metadata": metadata,
        }
        record["question"] = record.pop("SpiderSynQuestion")
        for key in ["SpiderQuestion"]:
            record.pop(key, None)

    with Path("./data/raw/spider_test_syn.json").open("w") as f:
        json.dump(dev_data, f, indent=2)

    with Path("data/raw/spider-realistic/spider-realistic.json").open() as f:
        dev_data = json.load(f)

    for record in tqdm(dev_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["schema"] = {
            "database": record.pop("db_id"),
            "metadata": metadata,
        }
        for key in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
            record.pop(key, None)

    with Path("./data/raw/spider_test_realistic.json").open("w") as f:
        json.dump(dev_data, f, indent=2)


def bird(path=Path("./data/raw/bird/")):
    databases = get_databases_info("bird")

    with (path / "train" / "train.json").open() as f:
        train_data = json.load(f)

    for record in tqdm(train_data[:]):
        db_id = record["db_id"]
        try:
            metadata = extract_metadata(record["SQL"], databases[db_id])
            record["schema"] = {
                "database": record.pop("db_id"),
                "metadata": metadata,
            }
            record["query"] = record.pop("SQL")
            for key in ["question_toks", "SQL_toks", "evidence_toks", "evidence"]:
                record.pop(key, None)
        except Exception:
            train_data.remove(record)

    with (path.parent / "bird_train.json").open("w") as f:
        json.dump(train_data, f, indent=2)

    with (path / "dev" / "dev.json").open() as f:
        dev_data = json.load(f)

    for record in tqdm(dev_data[:]):
        db_id = record["db_id"] if record["db_id"] != "movies_4" else "movie_4"
        try:
            metadata = extract_metadata(record["SQL"], databases[db_id])
            record["schema"] = {
                "database": record.pop("db_id"),
                "metadata": metadata,
            }
            record["query"] = record.pop("SQL")
            for key in [
                "question_toks",
                "SQL_toks",
                "evidence_toks",
                "evidence",
                "difficulty",
            ]:
                record.pop(key, None)
        except Exception:
            dev_data.remove(record)

    with (path.parent / "bird_test.json").open("w") as f:
        json.dump(dev_data, f, indent=2)

    with (path.parent / "bird_schemas.json").open("w") as f:
        json.dump(databases, f, indent=2)


if __name__ == "__main__":
    spider()
    spider_variants()
    bird()
