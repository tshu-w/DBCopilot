import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import networkx as nx
from transformers import PreTrainedTokenizer

from src.utils.helpers import Node, snode


class Trie:
    def __init__(self, end="_end"):
        self._root = Trie._nested_dict()
        self._end = end

    @staticmethod
    def _nested_dict():
        return defaultdict(Trie._nested_dict)

    def add(self, key, value=0):
        assert value is not None
        node = self._root
        for k in key:
            node = node[k]

        node[self._end] = value

    def _traverse(self, key):
        node = self._root
        for k in key:
            if k not in node:
                return None
            node = node[k]

        return node

    def search(self, key):
        node = self._traverse(key)
        return node.get(self._end, None) if node else None

    def findnext(self, key):
        node = self._traverse(key)
        return list(node) if node else []

    def delete(self, key):
        def _delete(node, key, depth=0):
            if depth == len(key):
                node.pop(self._end, None)
                return len(node) == 0

            k = key[depth]
            if k in node and _delete(node[k], key, depth + 1):
                node.pop(k)
                return len(node) == 0

            return False

        _delete(self._root, key)

    def __repr__(self):
        return json.dumps(self._root, indent=2)


@dataclass
class ConstraintDecoder:
    tokenizer: PreTrainedTokenizer
    G: nx.DiGraph

    def __post_init__(self):
        self.trie = Trie(end=self.tokenizer.sep_token_id)
        self._text2ids = {}

    def encode(self, text):
        if text not in self._text2ids:
            self._text2ids[text] = self.tokenizer.encode(text, add_special_tokens=False)

        return self._text2ids[text]

    def get_exist_schemas(self, sent):
        database, tables = None, []
        left = 0
        while left < len(sent):
            if sent[left] not in [
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]:
                right = left + 1
                while right < len(sent) and sent[right] != self.tokenizer.sep_token_id:
                    right += 1

                if right == len(sent):
                    break

                ids = sent[left:right]
                if database is None:
                    database = self.tokenizer.decode(ids)
                else:
                    table = self.tokenizer.decode(ids)
                    assert table is not None
                    tables.append(table)

                left = right
            else:
                left += 1

        return database, tables

    @contextmanager
    def temporary_add(self, prefix, exist_schemas):
        database, tables = exist_schemas
        if database is None:
            for node in self.G[snode]:
                self.trie.add(self.encode(node.name))
            yield self.trie.findnext(prefix)
            for node in self.G[snode]:
                self.trie.delete(self.encode(node.name))
        elif tables == []:
            dnode = Node(database, "source")
            for node in self.G[dnode]:
                self.trie.add(self.encode(node.name))
            yield self.trie.findnext(prefix)
            for node in self.G[dnode]:
                self.trie.delete(self.encode(node.name))
        else:
            remained = 0
            for table in tables:
                tnode = Node(table, database)
                for node in self.G[tnode]:
                    if node.name not in tables:
                        self.trie.add(self.encode(node.name))
                        remained += 1
            nxt = self.trie.findnext(prefix)
            if remained == 1 and nxt == [self.tokenizer.sep_token_id]:
                nxt = [self.tokenizer.eos_token_id]
            yield nxt
            remained = 0
            for table in tables:
                tnode = Node(table, database)
                for node in self.G[tnode]:
                    if node.name not in tables:
                        self.trie.delete(self.encode(node.name))

    def __call__(self, sent: list[int]) -> list[int]:
        # <database_name> <table_name_1> <table_name_2> <table_name_3>
        exist_schemas = self.get_exist_schemas(sent)
        prefix = []
        for i, token_id in reversed(list(enumerate(sent))):
            if token_id in [
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]:
                prefix = sent[i + 1 :]
                break

        with self.temporary_add(prefix, exist_schemas) as candidates:
            if self.tokenizer.sep_token_id in candidates:
                candidates.append(self.tokenizer.eos_token_id)
            return candidates
