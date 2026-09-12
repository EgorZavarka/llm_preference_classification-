
import json
import random

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


TEXT_COLUMNS = ["prompt", "response_a", "response_b"]


def parse_turns(value):
    turns = json.loads(value)

    if not isinstance(turns, list):
        raise ValueError("Expected a JSON list")

    if any(
        turn is not None and not isinstance(turn, str)
        for turn in turns
    ):
        raise ValueError("Unexpected turn type")

    return ["" if turn is None else turn for turn in turns]


def prepare_frame(df):
    result = df.copy()

    for column in TEXT_COLUMNS:
        turns = result[column].map(parse_turns)

        # Маркеры сохраняют хотя бы явные границы реплик.
        result[f"{column}_text"] = turns.map(
            lambda items: "\n\n".join(
                f"Turn {i + 1}: {text}"
                for i, text in enumerate(items)
            )
        )

        if column == "prompt":
            # Совпадает с логикой групп baseline.
            result["prompt_group"] = turns.map(
                lambda items: json.dumps(
                    [" ".join(text.split()) for text in items],
                    ensure_ascii=False,
                )
            )

    return result


def encode_frame(
    df,
    tokenizer,
    prompt_tokens,
    answer_tokens,
    batch_size=256,
):
    encoded = {}
    limits = {
        "prompt": prompt_tokens,
        "response_a": answer_tokens,
        "response_b": answer_tokens,
    }

    for column in TEXT_COLUMNS:
        texts = df[f"{column}_text"].tolist()
        sequences = []

        for left in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[left:left + batch_size],
                add_special_tokens=False,
                truncation=True,
                max_length=limits[column],
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            sequences.extend(batch["input_ids"])

        encoded[column] = sequences

    return encoded


class PreferenceDataset(Dataset):
    def __init__(
        self,
        encoded,
        cls_token_id,
        sep_token_id,
        labels=None,
        random_swap=False,
        fixed_swap=False,
    ):
        self.encoded = encoded
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.labels = (
            None
            if labels is None
            else np.asarray(labels, dtype=np.int64)
        )
        self.random_swap = random_swap
        self.fixed_swap = fixed_swap

    def __len__(self):
        return len(self.encoded["prompt"])

    def __getitem__(self, index):
        prompt = self.encoded["prompt"][index]
        answer_a = self.encoded["response_a"][index]
        answer_b = self.encoded["response_b"][index]

        swap = self.fixed_swap or (
            self.random_swap and random.random() < 0.5
        )

        if swap:
            answer_a, answer_b = answer_b, answer_a

        input_ids = (
            [self.cls_token_id]
            + prompt
            + [self.sep_token_id]
            + answer_a
            + [self.sep_token_id]
            + answer_b
            + [self.sep_token_id]
        )

        item = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

        if self.labels is not None:
            label = int(self.labels[index])

            if swap:
                label = [1, 0, 2][label]

            item["labels"] = label

        return item
