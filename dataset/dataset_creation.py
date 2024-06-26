import os
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from typing import Tuple, Dict
from sklearn.datasets import make_circles
import numpy as np


def get_dummy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    res = make_circles(n_samples=50, shuffle=True, noise=0.1, random_state=7, factor=0.6)
    for i, ex in enumerate(res[1]):
        res[1][i] = res[1][i] if res[1][i] == 1 else -1
    return res


"""
tweet_eval has 3 splits:
    train (45615)
    test (12284)
    validation (2000)

Every entry of the split is formed as: (text, label)
text: a string feature containing the tweet.
label: an int classification label with the following mapping:
    0: negative
    1: neutral
    2: positive
"""


def load_tweet_eval() -> DatasetDict:
    dataset = load_dataset("tweet_eval", "sentiment")
    dataset['train'] = concatenate_datasets([dataset['train'], dataset['validation']])
    del dataset['validation']

    return dataset


def to_binary(dataset: DatasetDict, splits: Tuple[str, str], seed: int) -> DatasetDict:
    """
    Discard data with neutral class (not relevant for binary classification)
    """
    for split in splits:
        dataset[split] = dataset[split].filter(lambda example: example['label'] != 1).shuffle(seed=seed)
    return dataset


def normalize_label(entry: Dict[str, str | int]) -> Dict[str, str | int]:
    entry['label'] -= 1
    return entry


def normalize(dataset: DatasetDict, splits: Tuple[str, str], seed: int) -> DatasetDict:
    """
    Unify the cardinality of elements in the positive class and in the negative class
    """
    for split in splits:
        new_dataset = []
        dataset[split] = dataset[split].map(normalize_label)
        pos, neg = (dataset[split].filter(lambda example: example['label'] == 1),
                    dataset[split].filter(lambda example: example['label'] == -1))
        target_dim = min(len(pos), len(neg))
        num = 0
        for p in pos:
            if num < target_dim:
                new_dataset.append(p)
                num += 1
        num = 0
        for n in neg:
            if num < target_dim:
                new_dataset.append(n)
                num += 1
        dataset[split] = Dataset.from_list(new_dataset).shuffle(seed=seed)
    return dataset


def save(dataset: DatasetDict, splits: Tuple[str, str], path: str) -> None:
    """
    Save dataset splits in data folder
    """
    os.makedirs(path, exist_ok=True)
    for split in splits:
        dataset[split].to_json(path + split + '.json')
