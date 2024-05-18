import itertools
import random
from collections import Counter, defaultdict, namedtuple
from collections.abc import Iterable, Iterator
from operator import attrgetter

import networkx as nx

Node = namedtuple("Node", ["name", "affiliation"])
snode = Node("source", "")


def chunks(lst: Iterable, n: int) -> Iterator[Iterable]:
    """Yield successive n-sized chunks from lst."""
    size = lst.shape[0] if hasattr(lst, "shape") else len(lst)
    for i in range(0, size, n):
        yield lst[i : i + n]


def schema2graph(schemas: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    for database, tables in schemas.items():
        links = defaultdict(set)
        dnode = Node(database, "source")
        G.add_edge(snode, dnode)
        for table in tables:
            tnode = Node(table["name"], database)
            G.add_edge(dnode, tnode)
            for column in table["columns"]:
                foreign_key = column.get("foreign_key", None)
                if foreign_key:
                    key = f'{foreign_key["table"]}.{foreign_key["column"]}'
                    links[key].update([foreign_key["table"], table["name"]])

        for tables in links.values():
            for tbl1, tbl2 in itertools.product(tables, tables):
                if tbl1 != tbl2:
                    tnode1 = Node(tbl1, database)
                    tnode2 = Node(tbl2, database)
                    G.add_edge(tnode1, tnode2)

    return G


def serialize_schema(
    schema: dict, G: nx.DiGraph, separator: str, shuffle: bool = True
) -> str:
    """
    Serialize a schema into a depth-first-search pre-order sequence of graph G.

    Input: {
      "database": "<database_name>",
      "metadata": [
        {"name": "<table_name_1>", "columns": ["<column_name_1>", "<column_name_2>"]},
        {"name": "<table_name_2>", "columns": ["<column_name_1>", "<column_name_2>", "<column_name_3>"]},
        {"name": "<table_name_3>", "columns": ["<column_name_1>"]}
      ]
    }

    Output: <database_name> <table_name_1> <table_name_2> <table_name_3>
    """
    # Check separator is not in labels
    assert separator not in schema["database"]
    assert all(separator not in t["name"] for t in schema["metadata"])

    nodes = {
        snode,
        Node(schema["database"], "source"),
        *[Node(t["name"], schema["database"]) for t in schema["metadata"]],
    }
    stack = [snode]
    visited = []
    while stack:
        node = stack.pop()
        visited.append(node)
        if set(visited) == nodes:
            break

        children = [
            child for child in list(G[node]) if child in nodes and child not in visited
        ]
        if shuffle:
            random.shuffle(children)
        stack.extend(children)
    else:
        print(schema)

    # ## Trail traversal
    # def dfs(node):
    #     visited = list(map(itemgetter(1), walk))
    #     if set(visited) == nodes:
    #         return visited

    #     children = [
    #         child
    #         for child in list(G[node])
    #         if child in nodes and (node, child) not in walk
    #     ]
    #     children = random.shuffle(children) if shuffle else children

    #     for child in children:
    #         walk.append((node, child))
    #         res = dfs(child)
    #         if res is not None:
    #             return res
    #         walk.pop()

    #     return None

    # nodes = {
    #     Node(schema["database"], "database"),
    #     *[
    #         Node(f'{schema["database"]}.{t["name"]}', "table")
    #         for t in schema["metadata"]
    #     ],
    # }
    # walk = []
    # visited = dfs(snode)
    # if visited is None:
    #     print(schema)
    #     breakpoint()

    return separator.join(map(attrgetter("name"), visited[1:]))


def deserialize_schema(s: str, separator: dict) -> dict:
    """
    Deserialize string of labels into a dict of database schema.

    Input: <database_name> <table_name_1> <table_name_2> <table_name_3>

    Output: {"database": "<database_name>", "tables": ["<table_name_1>", "<table_name_2>", "<table_name_3>"]}
    """
    # Remove space after separator for T5,
    # see https://github.com/huggingface/transformers/issues/24743
    s = s.replace(f"{separator} ", f"{separator}")

    database, *tables = s.split(f"{separator}")
    schema = {"database": database, "tables": tables}

    return schema


def stringize_schema(schema: dict) -> dict:
    """
    Stringize a dict of database schema.

    Input: {
      "database": "<database_name>",
      "metadata": [
        {"name": "<table_name_1>", "columns": ["<column_name_1>", "<column_name_2>"]},
        {"name": "<table_name_2>", "columns": ["<column_name_1>", "<column_name_2>", "<column_name_3>"]},
        {"name": "<table_name_3>", "columns": ["<column_name_1>"]}
      ]
    }

    Output:
    <database_name>
    - <table_name_1> (<column_name_1>, <column_name_2>)
    - <table_name_2> (<column_name_1>, <column_name_2>, <column_name_3>)
    - <table_name_3> (<column_name_1>)
    """
    tables = "\n".join(
        f"- {t['name']} ({', '.join(t['columns'])})" for t in schema["metadata"]
    )
    return f"{schema['database']}\n{tables}"


def schema2label(schema: dict, separator: str, shuffle: bool = True) -> str:
    """
    Transform a dict of database schema into a string of labels.

    Input: {
      "database": "<database_name>",
      "metadata": [
        {"name": "<table_name_1>", "columns": ["<column_name_1>", "<column_name_2>"]},
        {"name": "<table_name_2>", "columns": ["<column_name_1>", "<column_name_2>", "<column_name_3>"]},
        {"name": "<table_name_3>", "columns": ["<column_name_1>"]}
      ]
    }

    Output: <table_name_1> <table_name_2> <table_name_3>
    """
    # Check separator is not in labels
    assert separator not in schema["database"]
    assert all(separator not in t["name"] for t in schema["metadata"])

    tables = [t["name"] for t in schema["metadata"]]
    if shuffle:
        random.shuffle(tables)
    tables = separator.join(tables)
    return tables


def label2schema(s: str, separator: str, tbl2db: dict) -> dict:
    """
    Transform a string of labels into a dict of database schema.

    Input: <table_name_1> <table_name_2> <table_name_3>

    Output: {"database": "<database_name>", "tables": ["<table_name_1>", "<table_name_2>", "<table_name_3>"]}
    """
    try:
        # Remove space after separator for T5,
        # see https://github.com/huggingface/transformers/issues/24743
        s = s.replace(f"{separator} ", f"{separator}")

        tables = s.split(f"{separator}")
        dbs = [db for tbl in tables if tbl in tbl2db for db in tbl2db[tbl]]
        database = Counter(dbs).most_common(1)[0][0]
        schema = {
            "database": database,
            "tables": tables,
        }

        return schema
    except Exception:
        return {"database": "", "tables": []}
