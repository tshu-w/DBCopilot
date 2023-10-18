import itertools
import json
from operator import itemgetter
from pathlib import Path
from typing import Literal

import numpy as np
from lightning.fabric.utilities.seed import seed_everything
from ranx import Qrels, Run, evaluate
from retriv import DenseRetriever, SparseRetriever
from rich.console import Console
from rich.table import Table

METRICS = ["recall@1", "recall@10", "recall@50"]


def generate_collection(schemas, resolution) -> dict[str, str]:
    for db, tables in schemas:
        if resolution == "column":
            for tbl in tables:
                for col in tbl["columns"]:
                    doc = {
                        "id": f"{db}.{tbl['name']}.{col['name']}",
                        "text": f"{db} {tbl['name']} {col['name']}",
                    }
                    yield doc
        elif resolution == "table":
            for tbl in tables:
                text = f"{db} {tbl['name']} {' '.join(map(itemgetter('name'), tbl['columns']))}"
                doc = {"id": f"{db}.{tbl['name']}", "text": text}
                yield doc
        elif resolution == "database":
            text = " ".join(
                f"{tbl['name']} {' '.join(map(itemgetter('name'), tbl['columns']))}"
                for tbl in tables
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
    data_name: str,
    test_name: str,
    resolution: Literal["database", "table", "column"],
    retriever_class: SparseRetriever | DenseRetriever = SparseRetriever,
    retriever_kwargs: dict | None = None,
    tune: bool = False,
):
    try:
        retriever = retriever_class.load(retriever_kwargs["index_name"])
    except FileNotFoundError:
        retriever = retriever_class(**retriever_kwargs)
        with Path(f"data/{data_name}/schemas.json").open() as f:
            schemas = json.load(f)

        retriever = retriever.index(
            generate_collection(schemas.items(), resolution),
            show_progress=True,
            callback=lambda x: x,
        )
        if hasattr(retriever, "relative_doc_lens") and not isinstance(
            retriever.relative_doc_lens, np.ndarray
        ):
            retriever.relative_doc_lens = np.array(retriever.relative_doc_lens).reshape(
                -1
            )

        if tune and retriever_class != DenseRetriever:
            train_path = Path("data") / data_name / "train.json"
            with train_path.open() as f:
                train = json.load(f)

            queries = [
                {"id": str(i), "text": it["question"]} for i, it in enumerate(train)
            ]
            qrels = generate_qrels(train, resolution)
            retriever.autotune(
                queries=queries,
                qrels=qrels,
            )

        retriever.save()

    test = []
    test_paths = (Path("data") / test_name).glob("test*.json")
    for pth in test_paths:
        with pth.open() as f:
            test.extend(json.load(f))

    queries = [{"id": str(i), "text": it["question"]} for i, it in enumerate(test)]
    results = retriever.bsearch(queries, show_progress=True, cutoff=100)

    qrels = Qrels(generate_qrels(test, resolution))
    run = Run(results)
    return evaluate(qrels, run, metrics=METRICS)


if __name__ == "__main__":
    console = Console()
    table = Table(*["Data", "Test Set", "Resolution", "Retriever", "Tuned", *METRICS])
    datasets = {
        "spider": ["spider", "spider_syn", "spider_realistic", "spider_dr"],
        "bird": ["bird"],
        "wikisql": ["wikisql"],
    }
    resolutions = [
        "database",
        "table",
    ]
    retriever_classes = [
        SparseRetriever,
        DenseRetriever,
    ]
    tunes = [
        False,
        True,
    ]
    default_model = "sentence-transformers/all-mpnet-base-v2"
    tuned_model = {
        "spider": "sentence-transformers/all-mpnet-base-v2",
        "bird": "sentence-transformers/all-mpnet-base-v2",
        "wikisql": "sentence-transformers/all-mpnet-base-v2",
    }

    for data, tests in datasets.items():
        for test in tests:
            for resolution, retriever_class, tune in itertools.product(
                resolutions, retriever_classes, tunes
            ):
                retriever_type = (
                    "sparse"
                    if isinstance(retriever_class, SparseRetriever)
                    else "dense"
                )
                retriever_kwargs = {
                    "index_name": f"{data}/{resolution}_{retriever_type}_{str(tune).lower()}.index"
                }
                if retriever_class == DenseRetriever:
                    retriever_kwargs = {
                        "model": default_model if tune else tuned_model[data],
                        "use_ann": False,
                        **retriever_kwargs,
                    }

                seed_everything(42)
                result = retrieve_schemas(
                    data,
                    test,
                    resolution,
                    retriever_class,
                    retriever_kwargs,
                    tune,
                )
                table.add_row(
                    data,
                    test,
                    resolution,
                    retriever_class.__name__,
                    str(tune),
                    *map(str, result.values()),
                )

    console.print(table)
