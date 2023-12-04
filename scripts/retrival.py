import itertools
import json
from collections import Counter
from operator import itemgetter
from pathlib import Path
from typing import Literal

import nltk
import numpy as np
from lightning.fabric.utilities.seed import seed_everything
from ranx import Qrels, Run, evaluate
from retriv import DenseRetriever, SparseRetriever
from rich.console import Console
from rich.table import Table

DB_METRICS = ["recall@1", "recall@5"]
TBL_METRICS = ["recall@5", "recall@10", "recall@15", "recall@20"]
METRICS = ["DR@1", "DR@5", "TR@5", "TR@10", "TR@15", "TR@20"]

nltk.download = lambda *args, **kwargs: None


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
    force: bool = True,
) -> SparseRetriever | DenseRetriever:
    try:
        if force:
            raise FileNotFoundError
        retriever = retriever_class.load(retriever_kwargs["index_name"])
    except FileNotFoundError:
        retriever = retriever_class(**retriever_kwargs)
        with Path(f"data/{data_name}/schemas.json").open() as f:
            schemas = json.load(f)

        retriever = retriever.index(
            generate_collection(schemas.items(), resolution),
            show_progress=True,
            callback=lambda x: x,
            use_gpu=True,
        )
        if hasattr(retriever, "relative_doc_lens") and not isinstance(
            retriever.relative_doc_lens, np.ndarray
        ):
            retriever.relative_doc_lens = np.array(retriever.relative_doc_lens).reshape(
                -1
            )

        if tune and retriever_class != DenseRetriever:
            train_path = Path("data") / data_name / "synthetic.json"
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


def retrieve_schemas(
    data_name: str,
    test_name: str,
    retriever_class: SparseRetriever | DenseRetriever = SparseRetriever,
    retriever_kwargs: dict | None = None,
    tune: bool = False,
):
    retriever = get_retriever(
        data_name=data_name,
        resolution="table",
        retriever_class=retriever_class,
        retriever_kwargs=retriever_kwargs,
        tune=tune,
    )

    test_path = Path("data") / test_name / "test.json"
    with test_path.open() as f:
        test = json.load(f)

    queries = [{"id": str(i), "text": it["question"]} for i, it in enumerate(test)]
    tbl_results = retriever.bsearch(
        queries, show_progress=True, cutoff=100, batch_size=64
    )

    db_results = {}
    for q_id, doc_scores in tbl_results.items():
        scores, count = Counter(), Counter()
        for doc_id, doc_score in doc_scores.items():
            db = doc_id.split(".")[0]
            scores[db] += doc_score
            count[db] += 1
        db_average_scores = {db: scores[db] / count[db] for db in scores}
        db_results[str(q_id)] = db_average_scores

    db_qrels = generate_qrels(test, resolution="database")
    tbl_qrels = generate_qrels(test, resolution="table")

    db_scores = evaluate(Qrels(db_qrels), Run(db_results), metrics=DB_METRICS)
    tbl_scores = evaluate(Qrels(tbl_qrels), Run(tbl_results), metrics=TBL_METRICS)

    for qid, it in enumerate(test):
        if db_results[str(qid)]:
            dbs, _scores = zip(
                *sorted(db_results[str(qid)].items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            )
            test[qid]["pred_schemas"] = [
                {
                    "database": db,
                    "tables": [
                        tbl.split(".")[1]
                        for tbl in tbl_results[str(qid)]
                        if tbl.split(".")[0] == db
                    ],
                }
                for db in dbs
            ]
        else:
            test[qid]["pred_schemas"] = []

    retriever_type = "sparse" if retriever_class == SparseRetriever else "dense"
    result_path = (
        Path("results")
        / "retrieval"
        / f"{test_name}_{retriever_type}_{str(tune).lower()}.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(test, f, indent=2)

    return [*db_scores.values(), *tbl_scores.values()]


if __name__ == "__main__":
    console = Console()
    table = Table(*["Data", "Test Set", "Retriever", "Tuned", *METRICS])
    datasets = {
        "spider": ["spider", "spider_syn", "spider_realistic", "spider_dr"],
        "bird": ["bird"],
        "fiben": ["fiben"],
    }
    retriever_classes = [
        SparseRetriever,
        DenseRetriever,
    ]
    tunes = [
        False,
        True,
    ]
    default_model = "./models/all-mpnet-base-v2"
    tuned_model = {
        "spider": "./results/fit/rosy-elevator-178/xpqo1p3h/checkpoints/model/",
        "bird": "./results/fit/colorful-frog-176/sk0or8fo/checkpoints/model",
        "fiben": "./results/fit/cool-glade-177/nh0tbwuh/checkpoints/model",
    }

    for data, tests in datasets.items():
        print(data)
        for test in tests:
            print("\t", test)
            for retriever_class, tune in itertools.product(retriever_classes, tunes):
                retriever_type = (
                    "sparse" if retriever_class == SparseRetriever else "dense"
                )
                print("\t\t", retriever_type)
                retriever_kwargs = {
                    "index_name": f"{data}/table_{retriever_type}_{str(tune).lower()}.index"
                }
                if retriever_class == DenseRetriever:
                    retriever_kwargs = {
                        "model": tuned_model[data] if tune else default_model,
                        "use_ann": False,
                        **retriever_kwargs,
                    }

                seed_everything(42)
                scores = retrieve_schemas(
                    data,
                    test,
                    retriever_class,
                    retriever_kwargs,
                    tune,
                )
                table.add_row(
                    data,
                    test,
                    retriever_class.__name__,
                    str(tune),
                    *map(lambda x: str(round(x * 100, 2)), scores),
                )
                console.print(table)
