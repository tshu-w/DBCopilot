import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

if __name__ == "__main__":
    console = Console()
    table = Table(
        *["Dataset", "Train", "Dev", "Test", "# DB", "# Tables/DB", "# Cols/table"],
    )
    for dataset in [
        "spider",
        "bird",
        "wikisql",
        "spider_syn",
        "spider_realistic",
        "spider_dr",
    ]:
        res = {}
        for split in ["train", "dev", "test"]:
            cnt = 0
            for pth in Path(f"data/{dataset}").glob(f"{split}*.json"):
                with pth.open() as f:
                    data = json.load(f)
                    cnt += len(data)
            if cnt > 0:
                res[split] = cnt
            else:
                res[split] = "-"
        with Path(f"data/{dataset}/schemas.json").open() as f:
            schemas = json.load(f)
        res["# DB"] = len(schemas)
        res["# tables/DB"] = sum(len(v) for v in schemas.values()) / len(schemas)
        res["# cols/table"] = sum(
            len(t["columns"]) for v in schemas.values() for t in v
        ) / sum(len(v) for v in schemas.values())
        table.add_row(dataset, *map(str, res.values()))

    console.print(table)
