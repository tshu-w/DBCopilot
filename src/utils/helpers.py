import random
from collections import Counter
from collections.abc import Iterable, Iterator


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    size = lst.shape[0] if hasattr(lst, "shape") else len(lst)
    for i in range(0, size, n):
        yield lst[i : i + n]


def schema2label(
    schema: dict, separator: str, add_db: bool = True, shuffle: bool = True
) -> str:
    """
    Transform a dict of database schema into a string of labels.

    Input: {
      "database": "<database_name>",
      "metadata": [
        {"name": "<table_name_1>", "columns": ["<column_name_1>", "<column_name_2>"]},
        {"name": "<table_name_2>", "columns": ["<column_name_1>", "<column_name_2>", "<column_name_3>"]},
        {"name": "<table_name_3>", "columns": ["<column_name_1>"]}
      ]
    }

    Output: (<database_name> )<table_name_1> <table_name_2> <table_name_3>
    """
    # Check separator is not in labels
    assert separator not in schema["database"]
    assert all(separator not in t["name"] for t in schema["metadata"])

    tables = [t["name"] for t in schema["metadata"]]
    if shuffle:
        random.shuffle(tables)
    tables = separator.join(tables)
    return f"{schema['database']}{separator}{tables}" if add_db else tables


def schema2desc(schema: dict) -> str:
    """
    Transform a dict of database schema into a description.

    Input: {
      "database": "<database_name>",
      "metadata": [
        {"name": "<table_name_1>", "columns": ["<column_name_1>", "<column_name_2>"]},
        {"name": "<table_name_2>", "columns": ["<column_name_1>", "<column_name_2>", "<column_name_3>"]},
        {"name": "<table_name_3>", "columns": ["<column_name_1>"]}
      ]
    }

    Output:
    <database_name>
    - <table_name_1> (<column_name_1>, <column_name_2>)
    - <table_name_2> (<column_name_1>, <column_name_2>, <column_name_3>)
    - <table_name_3> (<column_name_1>)
    """
    tables = "\n".join(
        f"  - {t['name']} ({', '.join(t['columns'])})" for t in schema["metadata"]
    )
    return f"{schema['database']}\n{tables}"


def label2schema(
    s: str, separator: dict, add_db: bool = True, tbl2db: dict = {}
) -> dict:
    """
    Transform a string of labels into a dict of database schema.

    Input: (<database_name> )<table_name_1> <table_name_2> <table_name_3>

    Output: {"<database_name>": ["<table_name_1>", "<table_name_2>", "<table_name_3>"]}
    """
    try:
        # Remove space after separator for t5,
        # see https://github.com/huggingface/transformers/issues/24743
        s = s.replace(f"{separator} ", f"{separator}")

        schema = {}
        if add_db:
            database, *tables = s.split(f"{separator}")
            schema[database] = list(tables)
        else:
            tables = s.split(f"{separator}")
            dbs = Counter(table for table in tables if table in tbl2db)
            database = dbs.most_common(1)[0][0]
            schema[database] = tables

        return schema
    except Exception:
        return {}
