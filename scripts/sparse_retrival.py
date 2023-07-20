import json
from pathlib import Path
from typing import Literal

from ranx import Qrels, Run, evaluate
from retriv import SparseRetriever


def preprocess_schema(schema) -> dict:
    db, tables = schema
    text = " ".join(f"{tbl['name']} {' '.join(tbl['columns'])}" for tbl in tables)
    text = f"{db} {text}"
    doc = {"id": db, "text": text}
    return doc


def retrieve_schemas(
    dataset: Literal["spider", "bird"],
):
    retriever = SparseRetriever(index_name=f"data/{dataset}.index")
    with Path(f"data/{dataset}_schemas.json").open() as f:
        schemas = json.load(f)

    retriever = retriever.index(
        schemas.items(),
        callback=preprocess_schema,
    )

    for path in Path("data").glob(f"{dataset}_test*.json"):
        with path.open() as f:
            dev = json.load(f)

        queries = [{"id": str(i), "text": it["question"]} for i, it in enumerate(dev)]
        qrels_dict = {str(i): {it["schema"]["database"]: 1} for i, it in enumerate(dev)}
        results = retriever.bsearch(queries)
        qrels = Qrels(qrels_dict)
        run = Run(results)
        print(
            path.stem,
            evaluate(qrels, run, metrics=["recall@1", "recall@5", "recall@10"]),
        )


if __name__ == "__main__":
    for dataset in ["spider", "bird"]:
        retrieve_schemas(dataset)
