from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
from datasets import DatasetDict

model = SentenceTransformer("all-mpnet-base-v2")


def sentence_bert(entry: Dict[str, str | List]) -> Dict[str, str | List]:
    entry['sentence_bert'] = model.encode(entry['text'])
    return entry


def add_sentence_embedding(dataset: DatasetDict, splits: Tuple[str, str]):
    for split in splits:
        dataset[split] = dataset[split].map(sentence_bert)
    return dataset
