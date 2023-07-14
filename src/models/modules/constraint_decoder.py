import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

from transformers import PreTrainedTokenizer


class Trie:
    def __init__(self, end: str = "_end"):
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
        return [k for k in node if k != self._end] if node else []

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
    delimiters: dict[str, str]
    schemas: dict[str, list[dict]]

    def __post_init__(self):
        self.delimiter_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v)
            for k, v in self.delimiters.items()
        }

        self._text2ids = {}
        self.tries = {
            "database": Trie(),
            "table": defaultdict(Trie),
            "column": defaultdict(Trie),
        }
        for database in self.schemas:
            self.tries["database"].add(self.encode(database), database)
            for table in self.schemas[database]:
                self.tries["table"][database].add(
                    self.encode(table["name"]), table["name"]
                )
                for column in table["columns"]:
                    self.tries["column"][f"{database}.{table['name']}"].add(
                        self.encode(column), column
                    )

    def encode(self, text):
        if text not in self._text2ids:
            self._text2ids[text] = self.tokenizer.encode(text, add_special_tokens=False)

        return self._text2ids[text]

    def get_exist_schemas(self, sent):
        database = None
        tables, columns = [], []
        left = 0
        last_table_end = None
        while left < len(sent):
            if sent[left] == self.delimiter_ids["initiator"]:
                right = left + 1
                while (
                    right < len(sent) and sent[right] != self.delimiter_ids["separator"]
                ):
                    right += 1

                if right == len(sent):
                    break

                ids = sent[left + 1 : right]
                if database is None:
                    database = self.tries["database"].search(ids)
                else:
                    table = self.tries["table"][database].search(ids)
                    assert table is not None
                    tables.append(table)
                    last_table_end = right

                left = right
            else:
                left += 1

        left = last_table_end or len(sent)
        while left < len(sent):
            if sent[left] == self.delimiter_ids["separator"]:
                right = left + 1
                while right < len(sent) and sent[right] not in [
                    self.delimiter_ids["terminator"],
                    self.delimiter_ids["separator"],
                ]:
                    right += 1

                if right == len(sent):
                    break

                key = f"{database}.{tables[-1]}"
                ids = sent[left + 1 : right]
                column = self.tries["column"][key].search(ids)
                if column is None:
                    print(self.tokenizer.decode(sent))
                    print(self.tokenizer.decode(sent[left + 1 : right]))
                    print(key)
                    print(column)
                assert column is not None
                columns.append(column)

                left = right
            else:
                left += 1

        return database, tables, columns

    @contextmanager
    def temporary_delete(self, trie_type, prefix, exist_schemas):
        database, tables, columns = exist_schemas
        if trie_type == "database":
            yield self.tries["database"].findnext(prefix)
        else:
            if trie_type == "table":
                key = database
                elements = tables
            elif trie_type == "column":
                key = f"{database}.{tables[-1]}"
                elements = columns

            for elem in elements:
                self.tries[trie_type][key].delete(self.encode(elem))
            yield self.tries[trie_type][key].findnext(prefix)
            for elem in elements:
                self.tries[trie_type][key].add(self.encode(elem), elem)

    def __call__(self, sent: list[int]) -> list[int]:
        # (<database_name> (<table_name_1> <column_name_1> <column_name_2>) (<table_name_2> <column_name_1> <column_name_2> <column_name_3>) (<table_name_3> <column_name_1>))
        last_token_id = sent[-1]
        if last_token_id == self.tokenizer.pad_token_id:
            return [self.delimiter_ids["initiator"]]

        elif last_token_id in self.delimiter_ids.values():
            delimiter_nums = {
                k: len([i for i in sent if i == v])
                for k, v in self.delimiter_ids.items()
            }
            assert 0 <= delimiter_nums["initiator"] - delimiter_nums["terminator"] <= 2

            if last_token_id == self.delimiter_ids["initiator"]:
                # start of the database label
                if delimiter_nums["initiator"] == 1:
                    return self.tries["database"].findnext([])
                # start of the table label
                else:
                    exist_schemas = self.get_exist_schemas(sent)
                    with self.temporary_delete(
                        "table", [], exist_schemas
                    ) as candidates:
                        return candidates

            elif last_token_id == self.delimiter_ids["separator"]:
                # start of the table
                if delimiter_nums["initiator"] - delimiter_nums["terminator"] == 1:
                    return [self.delimiter_ids["initiator"]]
                # start of the column label
                elif delimiter_nums["initiator"] - delimiter_nums["terminator"] == 2:
                    exist_schemas = self.get_exist_schemas(sent)
                    with self.temporary_delete(
                        "column", [], exist_schemas
                    ) as candidates:
                        return candidates

            elif last_token_id == self.delimiter_ids["terminator"]:
                # end of the schema
                if delimiter_nums["initiator"] == delimiter_nums["terminator"]:
                    return [self.tokenizer.eos_token_id]
                # end of the column
                else:
                    exist_schemas = self.get_exist_schemas(sent)
                    with self.temporary_delete(
                        "table", [], exist_schemas
                    ) as candidates:
                        if candidates:
                            return [
                                self.delimiter_ids["separator"],
                                self.delimiter_ids["terminator"],
                            ]
                        else:
                            return [self.delimiter_ids["terminator"]]

        else:
            # in the middle of labels
            for i, token_id in reversed(list(enumerate(sent))):
                if token_id in self.delimiter_ids.values():
                    last_delimiter_id = token_id
                    last_delimiter_idx = i
                    break

            if last_delimiter_id == self.delimiter_ids["initiator"]:
                prefix = sent[last_delimiter_idx + 1 :]
                exist_schemas = self.get_exist_schemas(sent)
                if exist_schemas[0] is None:
                    with self.temporary_delete(
                        "database", prefix, exist_schemas
                    ) as candidates:
                        allow_delimiters = [self.delimiter_ids["separator"]]
                        return candidates if candidates else allow_delimiters
                else:
                    with self.temporary_delete(
                        "table", prefix, exist_schemas
                    ) as candidates:
                        allow_delimiters = [self.delimiter_ids["separator"]]
                        return candidates if candidates else allow_delimiters

            elif last_delimiter_id == self.delimiter_ids["separator"]:
                prefix = sent[last_delimiter_idx + 1 :]
                exist_schemas = self.get_exist_schemas(sent)
                with self.temporary_delete(
                    "column", prefix, exist_schemas
                ) as candidates:
                    database, tables, _ = exist_schemas
                    key = f"{database}.{tables[-1]}"
                    if self.tries["column"][key].search([]):
                        allow_delimiters = [
                            self.delimiter_ids["separator"],
                            self.delimiter_ids["terminator"],
                        ]
                    else:
                        allow_delimiters = [self.delimiter_ids["terminator"]]
                    return candidates if candidates else allow_delimiters

            elif last_delimiter_id == self.delimiter_ids["terminator"]:
                return []
