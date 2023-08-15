from typing import Iterable, Iterator


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    size = lst.shape[0] if hasattr(lst, "shape") else len(lst)
    for i in range(0, size, n):
        yield lst[i : i + n]


def str2schema(s: str, delimiters: dict) -> dict:
    """
    Converts string representation of a database schema into nested dictionary.

    Input: '(<database_name> (<table_name_1> <column_name_1> <column_name_2>) (<table_name_2> <column_name_1> <column_name_2> <column_name_3>) (<table_name_3> <column_name_1>))'

    Output: {
        "<database_name>": {
            "<table_name_1>": ["<column_name_1>", "<column_name_2>"],
            "<table_name_2>": ["<column_name_1>", "<column_name_2>", "<column_name_3>"],
            "<table_name_3>": ["<column_name_1>"]
        }
    }
    """

    initiator = delimiters["initiator"]
    separator = delimiters["separator"]
    terminator = delimiters["terminator"]
    try:
        # Remove space after delimiters for t5,
        # see https://github.com/huggingface/transformers/issues/24743
        for token in delimiters.values():
            s = s.replace(f"{token} ", f"{token}")

        schema = {}
        trimmed_str = s[len(initiator) : -len(terminator)]
        database, tables = trimmed_str.split(separator, 1)
        tables = tables[len(initiator) : -len(terminator)]
        tables = tables.split(f"{terminator}{separator}{initiator}")
        schema[database] = {}
        for table in tables:
            table_name, *columns = table.split(separator)
            schema[database][table_name] = columns

        return schema

    except Exception:
        return {}


def schema2str(schema: dict, delimiters: dict) -> str:
    """
    Converts dict of a database schema into string representation.

    Input: {
      "database": "<database_name>",
      "metadata": [
        {"name": "<table_name_1>", "columns": ["<column_name_1>", "<column_name_2>"]},
        {"name": "<table_name_2>", "columns": ["<column_name_1>", "<column_name_2>", "<column_name_3>"]},
        {"name": "<table_name_3>", "columns": ["<column_name_1>"]}
      ]
    }

    Output: '(<database_name> (<table_name_1> <column_name_1> <column_name_2>) (<table_name_2> <column_name_1> <column_name_2> <column_name_3>) (<table_name_3> <column_name_1>))'
    """
    initiator = delimiters["initiator"]
    separator = delimiters["separator"]
    terminator = delimiters["terminator"]

    for token in delimiters.values():
        assert token not in schema["database"]
        assert all(token not in t["name"] for t in schema["metadata"])
        assert all(
            token not in column for t in schema["metadata"] for column in t["columns"]
        )

    tables = separator.join(
        f"{initiator}{t['name']}{separator}{separator.join(t['columns'])}{terminator}"
        for t in schema["metadata"]
    )
    return f"{initiator}{schema['database']}{separator}{tables}{terminator}"
