import sqlite3
from pathlib import Path

import pandas as pd


def db2dfs(db_path: Path) -> dict[str, pd.DataFrame]:
    "Convert a SQLite databse to a dictionary of pandas Dataframes, with table names as keys."
    dataframes = {}
    con = sqlite3.connect(db_path)

    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [t[0] for t in tables]

    for table in tables:
        dataframes[table] = pd.read_sql(f"SELECT * from {table}", con)

    con.close()

    return dataframes


def annotate_db(db_dir: Path):
    ...


# fmt: off
if __name__ == "__main__":
    ...
