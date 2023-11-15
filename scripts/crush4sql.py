import asyncio
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import Literal

import guidance
import nltk
import numpy as np
from lightning.fabric.utilities.seed import seed_everything
from ranx import Qrels, Run, evaluate
from retriv import DenseRetriever, SparseRetriever
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm, tqdm_asyncio

sys.path.append(str(Path(__file__).parents[1]))

from src.utils import openai_with_usage  # noqa: F401

nltk.download = lambda *args, **kwargs: None

HALLUCINATED_PROMPT = """Hallucinate the minimal schema of a relational database that can be used to answer the natural language question. Here are some examples:

Example 1:

Question: What are the names of all stations with a latitude smaller than 37.5?

Tables:
1:station(name, latitude)

Example 2:

Question: Count the number of members in club Bootup Balitmore whose age is above 18.

Tables:
1:club(name, club id, club description, location)
2:member_of_club(club id, student id)
3:student(student id, age)

Example 3:

Question: Show the season, the player, and the name of the team that players belong to.

Tables:
1:match_season(season, team, player)
2:team(name, team identifier)

Example 4:

Question: Find the first name and age of the students who are playing both Football and Lacrosse.

Tables:
1:student(first name, age, student id)
2:sportsinfo(student id, sportname)

Example 5:

Question: What are the names of tourist attractions that can be reached by bus or is at address 254 Ottilie Junction?

Tables:
1:locations(address, location id)
2:tourist_attractions(location id, name, how to get there)

Example 6:

Question: Give the name of the highest paid instructor.

Tables:
1:instructor(name, salary)

Example 7:

Question: {{question}}

Tables: {{~gen "tables" temperature=0 max_tokens=500 top_p=1 frequency_penalty=0 presence_penalty=0}}
"""

DB_METRICS = ["recall@1", "recall@5"]
TBL_METRICS = ["recall@5", "recall@10", "recall@15", "recall@20"]
METRICS = ["DR@1", "DR@5", "TR@5", "TR@10", "TR@15", "TR@20"]


def extract_items(segment):
    # Check if the string matches the pattern word1(word2, word3)
    pattern = r"(\w+)\(([\w\s,]+)\)"
    match = re.match(pattern, segment)

    if match:
        word1 = match.group(1).replace(" ", "_")
        words = match.group(2).split(", ")
        return [f"{word1}.{word.replace(' ', '_')}" for word in words]
    else:
        return None


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
async def get_hallucinated_segments(
    question: str,
    model: guidance.llms.LLM = guidance.llms.OpenAI("gpt-3.5-turbo-instruct"),
):
    program = guidance(HALLUCINATED_PROMPT, llm=model, async_mode=True, silent=True)
    response = await program(question=question)
    try:
        segments = [l for l in response["tables"].splitlines() if l != ""]
        segments = [segment.split(":")[1].strip() for segment in segments]
        segments = [segment.replace("/", " ").replace("-", " ") for segment in segments]
        segments = [
            schema_item
            for segment in segments
            for schema_item in extract_items(segment)
        ]
        segments = [segment for segment in segments if "." in segment]
        return segments
    except Exception:
        return [response["tables"]]


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await tqdm_asyncio.gather(*(sem_coro(c) for c in coros))


# Hide program executor tracebacks
# https://github.com/guidance-ai/guidance/issues/412
async def program_executor_run(self, llm_session):
    """Execute the program."""
    self.llm_session = llm_session
    # first parse all the whitespace control
    # self.whitespace_control_visit(self.parse_tree)

    # now execute the program
    self.program._variables["@raw_prefix"] = ""
    await self.visit(
        self.parse_tree,
        guidance._variable_stack.VariableStack([self.program._variables], self),
    )


guidance._program_executor.ProgramExecutor.run = program_executor_run


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
    force: bool = False,
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


def greedy_select(segments, docs, BUDGET):
    """
    segments: list of segments
    docs: sorted list of dicts, where each dict is of the form
          {
            'doc_name': ...,
            'score': ...,           # final score
            'segment1': ...,        # score for segment1
            'segment2': ...,        # score for segment2
            ...
          }
    """
    APPLY_MEAN_ENTROPY = True
    NORMALIZE_cos = True

    def get_entropy(segment, cos):
        q_distri = cos[segment]
        q_distri = [(score + 1) / 2 for score in q_distri]
        sum_q_distri = sum(q_distri)
        q_distri = [score / sum_q_distri for score in q_distri]

        entropy = -sum([p * math.log(p) for p in q_distri])

        return entropy

    def get_snm(segment, n, n_idx, cos, selected, mean_entropy):
        if n in selected:
            return 0
        entropy = get_entropy(segment, cos)

        if APPLY_MEAN_ENTROPY:
            inv_entropy = mean_entropy - entropy
        else:
            inv_entropy = -entropy

        weight = 1 / (1 + math.exp(-inv_entropy))

        if NORMALIZE_cos:
            return weight * (cos[segment][n_idx] + 1) / 2
        else:
            return weight * cos[segment][n_idx]

    def get_complete_score(segment, n, n_idx, cos, selected, mean_entropy):
        snm = get_snm(segment, n, n_idx, cos, selected, mean_entropy)
        return (snm, n)

    def get_data(docs, segments):
        cos = defaultdict(list)
        schema_items = []
        for doc in docs:
            schema_items.append(doc["doc_name"])
            for segment in segments:
                cos[segment].append(doc[segment])
        return cos, schema_items

    cos, schema_items = get_data(docs, segments)

    list_entropies = []
    for segment in segments:
        list_entropies.append(get_entropy(segment, cos))
    mean_entropy = sum(list_entropies) / len(list_entropies)

    M = len(segments)
    covered = [0 for _ in range(M)]
    selected = {}
    while len(selected) < min(BUDGET, len(docs)):
        if (
            sum(covered) == M
        ):  # reset covered if BUDGET is not reached and all segments are covered
            covered = [0 for _ in range(M)]

        lst = []
        for i, segment in enumerate(segments):
            if covered[i] == 1:
                continue

            lst_score = []  # list of (score, n) tuples
            for n_idx, n in enumerate(schema_items):
                lst_score.append(
                    get_complete_score(segment, n, n_idx, cos, selected, mean_entropy)
                )

            s_dash, n_dash = max(lst_score, key=lambda x: x[0])
            best_n = (s_dash, n_dash)

            lst.append((i, best_n))
        i_dash, (s_dash, n_dash) = max(lst, key=lambda x: x[1][0])
        covered[i_dash] = 1
        selected[n_dash] = s_dash

    return selected


def process_seg_results(args):
    qid, qsegments, seg_results = args
    scored_docs = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for sid, segment in enumerate(qsegments):
        key = f"{qid}.{sid}"
        scores = seg_results[key]
        for doc, score in scores.items():
            scored_docs[doc]["score"] = max(scored_docs[doc]["score"], score)
            scored_docs[doc][segment] = score

    for segment in qsegments:
        for k, v in scored_docs.items():
            if segment not in scored_docs[k]:
                scored_docs[k][segment] = 0

    scored_docs = [{"doc_name": k, **v} for k, v in scored_docs.items()]
    greedy_docs = greedy_select(qsegments, scored_docs, BUDGET=20)
    return str(qid), greedy_docs


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

    model = guidance.llms.OpenAI("gpt-3.5-turbo-instruct")
    segments = asyncio.run(
        gather_with_concurrency(
            16, *[get_hallucinated_segments(it["question"], model) for it in test]
        )
    )
    print(model.usage)
    print(model.get_usage_cost_usd())

    seg_queries = [
        {"id": f"{qid}.{sid}", "text": f"{test[qid]['question']} {segment}"}
        for qid, qsegments in enumerate(segments)
        for sid, segment in enumerate(qsegments)
    ]
    seg_results = retriever.bsearch(seg_queries, show_progress=True, cutoff=100)

    tasks = ((qid, qsegments, seg_results) for qid, qsegments in enumerate(segments))
    with Pool() as pool:
        tbl_results = dict(
            tqdm(
                pool.imap(process_seg_results, tasks, chunksize=16), total=len(segments)
            )
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
        / f"crush4sql_{test_name}_{retriever_type}_{str(tune).lower()}.json"
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
        # True,
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
