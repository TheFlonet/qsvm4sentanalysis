from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from util.visualization import plot_confusion_matrix
import numpy as np


def evaluate(labels_test: np.ndarray, predictions: np.ndarray) -> None:
    print("Accuracy:", accuracy_score(labels_test, predictions))
    print("Precision:", precision_score(labels_test, predictions))
    print("Recall:", recall_score(labels_test, predictions))
    print("F1-score:", f1_score(labels_test, predictions))
    plot_confusion_matrix(confusion_matrix(labels_test, predictions))
