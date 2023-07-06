import json
import sys
from pathlib import Path

from sqlglot import exp, parse_one
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))


def extract_metadata(sql_query: str, database: list[dict]):
    parsed_query = parse_one(sql_query, read="sqlite")

    tables = [t.name.lower() for t in parsed_query.find_all(exp.Table)]
    columns = [c.name.lower() for c in parsed_query.find_all(exp.Column)]

    metadata = [
        {"name": t["name"], "columns": list(t["columns"].intersection(set(columns)))}
        for t in database
        if t["name"] in tables
    ]

    return metadata


def spider(dir=Path("./data/spider/")):
    with (dir / "tables.json").open() as f:
        raw_tables = json.load(f)

    databases = {}
    for db in raw_tables:
        tables = []
        for i, table_name in enumerate(db["table_names_original"]):
            columns = set(
                map(
                    lambda x: x[1].lower(),
                    filter(lambda x: x[0] == i, db["column_names_original"]),
                )
            )
            tables.append({"name": table_name.lower(), "columns": columns})

        databases[db["db_id"]] = tables

    train_files = list(dir.glob("train_*.json"))
    train_data = []
    for f in train_files:
        with f.open() as f:
            train_data.extend(json.load(f))

    for record in tqdm(train_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["metadata"] = metadata
        record["database"] = record.pop("db_id")
        for key in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
            record.pop(key, None)

    with (dir.parent / "train_spider.json").open("w") as f:
        json.dump(train_data, f, indent=2, sort_keys=True)

    with (dir / "dev.json").open() as f:
        dev_data = json.load(f)

    for record in tqdm(dev_data):
        metadata = extract_metadata(record["query"], databases[record["db_id"]])
        record["metadata"] = metadata
        record["database"] = record.pop("db_id")
        for key in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
            record.pop(key, None)

    with (dir.parent / "dev_spider.json").open("w") as f:
        json.dump(train_data, f, indent=2, sort_keys=True)


def bird(dir=Path("./data/bird/")):
    with (dir / "train" / "train_tables.json").open() as f:
        raw_tables = json.load(f)

    with (dir / "dev" / "dev_tables.json").open() as f:
        raw_tables.extend(json.load(f))

    databases = {}
    for db in raw_tables:
        tables = []
        for i, table_name in enumerate(db["table_names_original"]):
            columns = set(
                map(
                    lambda x: x[1].lower(),
                    filter(lambda x: x[0] == i, db["column_names_original"]),
                )
            )
            tables.append({"name": table_name.lower(), "columns": columns})

        databases[db["db_id"]] = tables

    with (dir / "train" / "train.json").open() as f:
        train_data = json.load(f)

    for record in tqdm(train_data[:]):
        db_id = record["db_id"] if record["db_id"] != "movies_4" else "movie_4"
        try:
            metadata = extract_metadata(record["SQL"], databases[db_id])
            record["metadata"] = metadata
            record["database"] = record.pop("db_id")
            record["query"] = record.pop("SQL")
            for key in ["question_toks", "SQL_toks", "evidence_toks", "evidence"]:
                record.pop(key, None)
        except Exception:
            train_data.remove(record)

    with (dir.parent / "train_bird.json").open("w") as f:
        json.dump(train_data, f, indent=2, sort_keys=True)

    with (dir / "dev" / "dev.json").open() as f:
        dev_data = json.load(f)

    for record in tqdm(dev_data[:]):
        db_id = record["db_id"] if record["db_id"] != "movies_4" else "movie_4"
        try:
            metadata = extract_metadata(record["SQL"], databases[db_id])
            record["metadata"] = metadata
            record["database"] = record.pop("db_id")
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

    with (dir.parent / "dev_bird.json").open("w") as f:
        json.dump(dev_data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    spider()
    bird()
