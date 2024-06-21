import logging
import os
import random
from typing import Tuple
from dotenv import load_dotenv
import datasets
from sklearn.svm import SVC
from scipy.special import softmax
from dataset.main import generate_data
from classical.CSVM import CSVM
from quantum.QSVM import QSVM
from util.evaluation import evaluate
from util.time_elapsed import eval_time
import numpy as np
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from util.visualization import plot_generalized_boundary

SHOW_MODE = False


@eval_time
def sklearn_test(examples_train: np.array, examples_test: np.array,
                 labels_train: np.array, labels_test: np.array) -> None:
    if SHOW_MODE:
        plot_generalized_boundary(SVC(kernel='rbf', gamma='auto'), examples_train, labels_train, 'Sk-Learn')
        return
    svm_model = SVC(kernel='rbf', gamma='auto')
    log.info('Training with sklearn'.upper())
    svm_model.fit(examples_train, labels_train)
    log.info('Predict with sklearn'.upper())
    predictions = svm_model.predict(examples_test)
    log.info('Testing with sklearn'.upper())
    evaluate(labels_test, predictions)


@eval_time
def cplex_test(examples_train: np.array, examples_test: np.array,
               labels_train: np.array, labels_test: np.array) -> None:
    if SHOW_MODE:
        plot_generalized_boundary(CSVM(big_c=255), examples_train, labels_train, 'CPLEX')
        return
    svm_model = CSVM(big_c=255)
    log.info('Training with cplex'.upper())
    svm_model.fit(examples_train, labels_train)
    log.info('Predict with cplex'.upper())
    predictions = svm_model.predict(examples_test)
    log.info('Testing with cplex'.upper())
    evaluate(labels_test, predictions)


@eval_time
def dwave_test(examples_train: np.array, examples_test: np.array,
               labels_train: np.array, labels_test: np.array) -> None:
    if SHOW_MODE:
        plot_generalized_boundary(QSVM(big_c=255), examples_train, labels_train, 'D-WAVE')
        return
    svm_model = QSVM(big_c=255)  # , lazy_loading_path='./remote')
    log.info('Training with d-wave'.upper())
    svm_model.fit(examples_train, labels_train)
    log.info('Predict with d-wave'.upper())
    predictions = svm_model.predict(examples_test)
    log.info('Testing with d-wave'.upper())
    evaluate(labels_test, predictions)


@eval_time
def transformer_test(examples: np.array, labels: np.array) -> None:
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    log.info('Loading model'.upper())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    log.info('Predict with Transformer'.upper())
    predictions = []
    for example in examples:
        output = model(**tokenizer(example, return_tensors='pt'))
        scores = softmax(output[0][0].detach().numpy())
        ranking = np.argsort(scores)[::-1]
        skip = False
        for i in range(scores.shape[0]):
            if skip:
                continue
            if ranking[i] == 2:  # Positive class
                predictions.append(1)
                skip = True
            elif ranking[i] == 0:  # Negative class
                predictions.append(-1)
                skip = True
    log.info('Testing with Transformer'.upper())
    evaluate(labels, np.array(predictions))


def resize_dataset(dataset: Dataset, size: int, seed: int) -> Dataset:
    pos = dataset.filter(lambda ex: ex['label'] == 1).shuffle(seed=seed).select(range(size // 2))
    neg = dataset.filter(lambda ex: ex['label'] == -1).shuffle(seed=seed).select(range(size // 2))
    return concatenate_datasets([pos, neg])


def get_data(seed: int) -> Tuple[Dataset, Dataset]:
    if not os.path.exists('./data') or not os.listdir('./data'):
        log.info('Missing data, dataset generation in progress'.upper())
        generate_data(seed)
    train = datasets.Dataset.from_json('./data/train.json')
    test = datasets.Dataset.from_json('./data/test.json')
    train, test = resize_dataset(train, 4096, seed), resize_dataset(test, 2048, seed)
    return train, test


def main() -> None:
    load_dotenv()
    log.info('Loading dataset'.upper())
    with open('util/1000.prime', 'r') as f:
        primes = random.choices([int(x.strip()) for x in f.read().split(',')], k=1)
    for prime in primes:
        log.info(f'Test with prime {prime}'.upper())
        train, test = get_data(prime)
        log.info(f'Train size: {len(train)}, Test size: {len(test)}'.upper())
        ex_train = np.array(train['sentence_bert'])
        l_train = np.array(train['label'])
        test_embedding = np.array(test['sentence_bert'])
        test_text = np.array(test['text'])
        l_test = np.array(test['label'])
        sklearn_test(ex_train, test_embedding, l_train, l_test)
        cplex_test(ex_train, test_embedding, l_train, l_test)
        dwave_test(ex_train, test_embedding, l_train, l_test)
        transformer_test(test_text, l_test)


if __name__ == '__main__':
    log = logging.getLogger('qsvm')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    log.addHandler(handler)
    file_handler = logging.FileHandler('complete_dataset.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    main()
