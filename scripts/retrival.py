import itertools
import json
from operator import itemgetter
from pathlib import Path
from typing import Literal

import numpy as np
from joblib import Memory
from lightning.fabric.utilities.seed import seed_everything
from ranx import Qrels, Run, evaluate
from retriv import DenseRetriever, SparseRetriever
from rich.console import Console
from rich.table import Table

METRICS = ["recall@1", "recall@5", "recall@10", "recall@25", "recall@50"]
memory = Memory(Path.home() / ".cache" / "joblib", verbose=0)


def generate_collection(schemas, resolution) -> dict[str, str]:
    for db, tables in schemas:
        if resolution == "column":
            for tbl in tables:
                for col in tbl["columns"]:
                    doc = {
                        "id": f"{db}.{tbl['name']}.{col['name']}",
                        "text": f"{db} {tbl['normalized_name']} {col['normalized_name']}",
                    }
                    yield doc
        elif resolution == "table":
            for tbl in tables:
                text = f"{db} {tbl['normalized_name']} {' '.join(map(itemgetter('normalized_name'), tbl['columns']))}"
                doc = {"id": f"{db}.{tbl['name']}", "text": text}
                yield doc
        elif resolution == "database":
            text = " ".join(
                f"{tbl['normalized_name']} {' '.join(map(itemgetter('normalized_name'), tbl['columns']))}"
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


def get_retriever(
    data_name: str,
    resolution: Literal["database", "table", "column"],
    retriever_class: SparseRetriever | DenseRetriever = SparseRetriever,
    retriever_kwargs: dict | None = None,
    tune: bool = False,
) -> SparseRetriever | DenseRetriever:
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
            if data_name == "wikisql":
                train_path == Path("data") / data_name / "dev.json"
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

    return retriever


@memory.cache
def get_qrels_and_results(
    data_name: str,
    test_name: str,
    resolution: Literal["database", "table", "column"],
    retriever_class: SparseRetriever | DenseRetriever = SparseRetriever,
    retriever_kwargs: dict | None = None,
    tune: bool = False,
):
    retriever = get_retriever(
        data_name=data_name,
        resolution=resolution,
        retriever_class=retriever_class,
        retriever_kwargs=retriever_kwargs,
        tune=tune,
    )

    test = []
    test_paths = (Path("data") / test_name).glob("test*.json")
    for pth in test_paths:
        with pth.open() as f:
            test.extend(json.load(f))

    queries = [{"id": str(i), "text": it["question"]} for i, it in enumerate(test)]
    results = retriever.bsearch(queries, show_progress=True, cutoff=100)

    qrels = generate_qrels(test, resolution)

    return qrels, results


def retrieve_schemas(
    data_name: str,
    test_name: str,
    resolution: Literal["database", "table", "column"],
    retriever_class: SparseRetriever | DenseRetriever = SparseRetriever,
    retriever_kwargs: dict | None = None,
    tune: bool = False,
):
    qrels, results = get_qrels_and_results(
        data_name=data_name,
        test_name=test_name,
        resolution=resolution,
        retriever_class=retriever_class,
        retriever_kwargs=retriever_kwargs,
        tune=tune,
    )
    return evaluate(Qrels(qrels), Run(results), metrics=METRICS)


if __name__ == "__main__":
    console = Console()
    table = Table(*["Data", "Test Set", "Resolution", "Retriever", "Tuned", *METRICS])
    datasets = {
        "spider": ["spider", "spider_syn", "spider_realistic", "spider_dr"],
        "bird": ["bird"],
        "fiben": ["fiben"],
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
        # True,
    ]
    default_model = "sentence-transformers/all-mpnet-base-v2"
    tuned_model = {
        "spider": "sentence-transformers/all-mpnet-base-v2",
        "bird": "sentence-transformers/all-mpnet-base-v2",
        "wikisql": "sentence-transformers/all-mpnet-base-v2",
        "fiben": "sentence-transformers/all-mpnet-base-v2",
    }

    for data, tests in datasets.items():
        print(data)
        for test in tests:
            print("\t", test)
            for resolution, retriever_class, tune in itertools.product(
                resolutions, retriever_classes, tunes
            ):
                retriever_type = (
                    "sparse" if retriever_class == SparseRetriever else "dense"
                )
                print("\t\t", resolution, retriever_type)
                retriever_kwargs = {
                    "index_name": f"{data}/{resolution}_{retriever_type}_{str(tune).lower()}.index"
                }
                if retriever_class == DenseRetriever:
                    retriever_kwargs = {
                        "model": tuned_model[data] if tune else default_model,
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
                    *map(lambda x: str(round(x * 100, 2)), result.values()),
                )

    console.print(table)
