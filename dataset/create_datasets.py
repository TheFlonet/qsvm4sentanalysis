from datasets import load_dataset, Dataset
import random
from typing import Tuple

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


def create_datasets() -> None:
    """
    Create and save dataset splits in data folder
    """
    dataset = load_dataset('tweet_eval', 'sentiment')
    for split_name, split_dataset in dataset.items():
        '''
        An initial cleaning operation is applied to each split in the dataset.
        This operation serves to:
            - Discard data with neutral class (not relevant for binary classification)
            - Unify the cardinality of elements in the positive class and that of elements in the negative class
        '''
        filtered_dataset = []
        pos_entry = []
        for entry in split_dataset:
            if entry['label'] == 0:
                filtered_dataset.append(entry)
            if entry['label'] == 2:
                pos_entry.append(entry)
        random.shuffle(pos_entry)
        pos_entry = pos_entry[:len(filtered_dataset)]
        filtered_dataset.extend(pos_entry)
        random.shuffle(filtered_dataset)
        Dataset.from_list(filtered_dataset).to_json(f'data/{split_name}.json')


def load_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets from json files in data folder
    :return: 3 dataset splits (train, test, validation)
    """
    return (Dataset.from_json('data/train.json'),
            Dataset.from_json('data/test.json'),
            Dataset.from_json('data/validation.json'))
