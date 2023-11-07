from dataclasses import dataclass
from typing import Literal

import networkx as nx
import torch
from transformers import PreTrainedTokenizer

from .helpers import schema2label, serialize_schema, stringize_schema


@dataclass
class Text2SchemaCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int | None = None
    relational: bool = True
    G: nx.Graph | None = None
    label_pad_token_id: int = -100

    def __call__(self, batch: list[dict]) -> dict[str, any]:
        inputs, targets = [], []
        for instance in batch:
            question, schema = instance["question"], instance["schema"]
            if self.relational:
                target = serialize_schema(
                    schema, self.G, separator=self.tokenizer.sep_token
                )
            else:
                target = schema2label(schema, separator=self.tokenizer.sep_token)
            targets.append(target)

            if question is None:
                inputs.append("")
            else:
                inputs.append(question)

        features = self.tokenizer(
            text=inputs,
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        features["labels"][
            features["labels"] == self.tokenizer.pad_token_id
        ] = self.label_pad_token_id
        return features


@dataclass
class Schema2TextCollator:
    tokenizer: PreTrainedTokenizer
    model_mode: Literal["seq2seq", "causal"]
    max_length: int | None = None
    prompt_template: str = """Ask a question that needs to be answered by combining the contents of all the tables, based on the database schema provided below.
{schema}
"""
    label_pad_token_id: int = -100

    def __call__(self, batch: list[dict]) -> dict[str, any]:
        inputs, targets = [], []
        for instance in batch:
            schema = stringize_schema(instance["schema"])
            inputs.append(self.prompt_template.format(schema=schema))

        if "question" in batch[0]:
            targets = [instance["question"] for instance in batch]
        else:
            targets = [""] * len(inputs)

        if self.model_mode == "seq2seq":
            features = self.tokenizer(
                text=inputs,
                text_target=targets,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        elif self.model_mode == "causal":
            targets = [
                f"{s}{t}{self.tokenizer.eos_token}" if t else f"{s}"
                for s, t in zip(inputs, targets)
            ]
            features = self.tokenizer(
                text=inputs,
                text_target=targets,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            source_lens = self.tokenizer(
                text=inputs,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
            )["length"]
            for label, source_len in zip(features["labels"], source_lens):
                label[:source_len] = torch.tensor([-100] * source_len)

        features["labels"][
            features["labels"] == self.tokenizer.pad_token_id
        ] = self.label_pad_token_id
        return features


@dataclass
class ContrastiveCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int | None = None

    def __call__(self, batch: list[dict]) -> dict[str, any]:
        questions, schemas = [], []
        for instance in batch:
            question, schema = instance["question"], instance["schema"]

            text = " ".join(
                f"{tbl['name']} {' '.join(tbl['columns'])}"
                for tbl in schema["metadata"]
            )

            questions.append(question)
            schemas.append(text)

        features1 = self.tokenizer(
            text=questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features2 = self.tokenizer(
            text=schemas,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return features1, features2
