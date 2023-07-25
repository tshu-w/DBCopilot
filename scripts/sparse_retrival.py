import json
from pathlib import Path
from typing import Literal

from ranx import Qrels, Run, evaluate
from retriv import SparseRetriever


def generate_collection(schemas, resolution) -> dict[str, str]:
    for db, tables in schemas:
        if resolution == "column":
            for tbl in tables:
                for col in tbl["columns"]:
                    doc = {
                        "id": f"{db}.{tbl['name']}.{col}",
                        "text": f"{db} {tbl['name']} {col}",
                    }
                    yield doc
        elif resolution == "table":
            for tbl in tables:
                text = f"{db} {tbl['name']} {' '.join(tbl['columns'])}"
                doc = {"id": f"{db}.{tbl['name']}", "text": text}
                yield doc
        elif resolution == "database":
            text = " ".join(
                f"{tbl['name']} {' '.join(tbl['columns'])}" for tbl in tables
            )
            text = f"{db} {text}"
            doc = {"id": db, "text": text}
            yield doc
        else:
            raise ValueError(f"Unknown resolution: {resolution}")


def generate_qrels(instances, resolution) -> dict[str, dict]:
    qrels = {}
    for i, it in enumerate(instances):
        if resolution == "column":
            qrels[str(i)] = {
                f'{it["schema"]["database"]}.{tbl["name"]}.{col}': 1
                for tbl in it["schema"]["metadata"]
                for col in tbl["columns"]
            }
        elif resolution == "table":
            qrels[str(i)] = {
                f'{it["schema"]["database"]}.{tbl["name"]}': 1
                for tbl in it["schema"]["metadata"]
            }
        elif resolution == "database":
            qrels[str(i)] = {it["schema"]["database"]: 1}
        else:
            raise ValueError(f"Unknown resolution: {resolution}")
    return qrels


def retrieve_schemas(
    dataset: Literal["spider", "bird"],
    resolution: Literal["database", "table", "column"],
):
    retriever = SparseRetriever(index_name=f"data/{dataset}.index")
    with Path(f"data/{dataset}_schemas.json").open() as f:
        schemas = json.load(f)

    retriever = retriever.index(
        generate_collection(schemas.items(), resolution),
        show_progress=False,
    )

    for path in Path("data").glob(f"{dataset}_test*.json"):
        with path.open() as f:
            test = json.load(f)

        queries = [{"id": str(i), "text": it["question"]} for i, it in enumerate(test)]
        results = retriever.bsearch(queries, show_progress=False)
        qrels = Qrels(generate_qrels(test, resolution))
        run = Run(results)
        metrics = ["recall@1", "recall@5", "recall@10"]
        print(
            path.stem,
            resolution,
            evaluate(qrels, run, metrics=metrics),
        )


if __name__ == "__main__":
    for dataset in ["spider", "bird"]:
        for resolution in ["database", "table", "column"]:
            retrieve_schemas(dataset, resolution)
