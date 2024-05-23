import json
from pathlib import Path

from rich import print
from scipy import stats

if __name__ == "__main__":
    datasets = {
        "spider": ["spider", "spider_syn", "spider_realistic"],
        "bird": ["bird"],
        "fiben": ["fiben"],
    }
    routing_dirs = {
        "spider": Path("./results/test/silver-sun-65/k5ou2ykv"),
        "bird": Path("./results/test/happy-dew-65/ey4tt3n2"),
        "fiben": Path("./results/test/electric-donkey-66/sj2wkc2b"),
    }

    for dataset, tests in datasets.items():
        for test in tests:
            metrics = {}
            for method in ["DBCopilot", "dpr", "crush_bm25"]:
                if method == "DBCopilot":
                    res_file = (
                        routing_dirs[dataset] / f'{test.replace(dataset, "test")}.json'
                    )
                elif method == "dpr":
                    res_file = Path("results") / "retrieval" / f"{test}_dense_true.json"
                elif method == "crush_bm25":
                    res_file = (
                        Path("results")
                        / "retrieval"
                        / f"crush4sql_{test}_sparse_false.json"
                    )

                with res_file.open("r") as f:
                    res = json.load(f)

                metrics[method] = []
                for it in res:
                    gold_dbs = [it["schema"]["database"]]
                    gold_tbls = [
                        f'{it["schema"]["database"]}.{metadata["name"]}'
                        for metadata in it["schema"]["metadata"]
                    ]
                    pred_dbs = [pred["database"] for pred in it["pred_schemas"]]
                    pred_tbls = [
                        f'{pred["database"]}.{tbl}'
                        for pred in it["pred_schemas"]
                        for tbl in pred["tables"]
                    ]
                    metric = {
                        "DR@1": len(set(gold_dbs) & set(pred_dbs[:1]))
                        / len(set(gold_dbs)),
                        "DR@5": len(set(gold_dbs) & set(pred_dbs[:5]))
                        / len(set(gold_dbs)),
                        "TR@5": len(set(gold_tbls) & set(pred_tbls[:5]))
                        / len(set(gold_tbls)),
                        "TR@15": len(set(gold_tbls) & set(pred_tbls[:15]))
                        / len(set(gold_tbls)),
                    }
                    metrics[method].append(metric)

            print(f"Dataset: {dataset}, Test: {test}")
            for metric_name in ["DR@1", "DR@5", "TR@5", "TR@15"]:
                print(f"Metric: {metric_name}")
                perf1 = [m[metric_name] for m in metrics["DBCopilot"]]
                perf2 = [m[metric_name] for m in metrics["dpr"]]
                print(f"DPR {stats.ttest_rel(perf1, perf2).pvalue:.2f}")
                perf1 = [m[metric_name] for m in metrics["DBCopilot"]]
                perf2 = [m[metric_name] for m in metrics["crush_bm25"]]
                print(f"CRUSH_BM25 {stats.ttest_rel(perf1, perf2).pvalue:.2f}")
