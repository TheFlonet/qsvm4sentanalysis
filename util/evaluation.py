import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

log = logging.getLogger('qsvm')


def evaluate(labels_test: np.ndarray, predictions: np.ndarray) -> None:
    log.info(f'Accuracy: {accuracy_score(labels_test, predictions):.2f}')
    log.info(f'Precision: {precision_score(labels_test, predictions):.2f}')
    log.info(f'Recall: {recall_score(labels_test, predictions):.2f}')
    log.info(f'F1-score: {f1_score(labels_test, predictions):.2f}')
    log.info(f'Confusion matrix [tn, fp, fn, tp]: {list(confusion_matrix(labels_test, predictions).ravel())}')
