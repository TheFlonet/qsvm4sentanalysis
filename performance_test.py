import logging
from typing import Tuple
from dotenv import load_dotenv
import datasets
from sklearn.svm import SVC
from classical.CSVM import CSVM
from quantum.QSVM import QSVM
from util.evaluation import evaluate
from util.time_elapsed import eval_time
import numpy as np
from datasets import concatenate_datasets, DatasetDict, Dataset


@eval_time
def sklearn_test(examples_train: np.array, examples_test: np.array,
                 labels_train: np.array, labels_test: np.array) -> None:
    svm_model = SVC(kernel='rbf', gamma='auto')
    log.info('Training with sklearn'.upper())
    svm_model.fit(examples_train, labels_train)
    log.info('Predict with sklearn'.upper())
    predictions = svm_model.predict(examples_test)
    log.info('Testing with sklearn'.upper())
    evaluate(labels_test, predictions)


@eval_time
def gurobi_test(examples_train: np.array, examples_test: np.array,
                labels_train: np.array, labels_test: np.array) -> None:
    svm_model = CSVM(big_c=255)
    log.info('Training with gurobi'.upper())
    svm_model.fit(examples_train, labels_train)
    log.info('Predict with gurobi'.upper())
    predictions = svm_model.predict(examples_test)
    log.info('Testing with gurobi'.upper())
    evaluate(labels_test, predictions)


@eval_time
def dwave_test(examples_train: np.array, examples_test: np.array,
               labels_train: np.array, labels_test: np.array) -> None:
    svm_model = QSVM(big_c=255)
    log.info('Training with d-wave'.upper())
    svm_model.fit(examples_train, labels_train)
    log.info('Predict with d-wave'.upper())
    predictions = svm_model.predict(examples_test)
    log.info('Testing with d-wave'.upper())
    evaluate(labels_test, predictions)


def resize_dataset(dataset: DatasetDict, size: int, seed: int) -> Dataset:
    pos = dataset.filter(lambda ex: ex['label'] == 1).shuffle(seed=seed).select(range(size // 2))
    neg = dataset.filter(lambda ex: ex['label'] == -1).shuffle(seed=seed).select(range(size // 2))
    return concatenate_datasets([pos, neg])


def get_data(seed: int) -> Tuple[Dataset, Dataset]:
    train = resize_dataset(datasets.DatasetDict.from_json('./data/train.json'), 4096, seed)
    test = resize_dataset(datasets.DatasetDict.from_json('./data/test.json'), 2048, seed)
    return train, test


def main() -> None:
    load_dotenv()
    log.info('Loading dataset'.upper())
    train, test = get_data(7)
    ex_train = np.array(train['sentence_bert'])
    l_train = np.array(train['label'])
    ex_test = np.array(test['sentence_bert'])
    l_test = np.array(test['label'])
    sklearn_test(ex_train, ex_test, l_train, l_test)
    gurobi_test(ex_train, ex_test, l_train, l_test)
    dwave_test(ex_train, ex_test, l_train, l_test)


if __name__ == '__main__':
    log = logging.getLogger('qsvm')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    main()
