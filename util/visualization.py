import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from classical.CSVM import CSVM
from quantum.QSVM import QSVM


def plot_decision_boundary(model: SVC | CSVM | QSVM, examples: np.ndarray, labels: np.ndarray, title) -> None:
    h = 0.02
    x_min, x_max = examples[:, 0].min() - 1, examples[:, 0].max() + 1
    y_min, y_max = examples[:, 1].min() - 1, examples[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('coolwarm'), alpha=0.8)
    plt.scatter(examples[:, 0], examples[:, 1], c=labels, cmap=plt.cm.get_cmap('coolwarm'), edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary')
    plt.savefig(f'img/decision_boundary_{title}.png')
    # plt.show()
    plt.clf()


def plot_dataset(examples: np.ndarray, labels: np.ndarray) -> None:
    plt.scatter(examples[:, 0], examples[:, 1], c=labels)
    plt.savefig('img/dataset.png')
    # plt.show()
    plt.clf()
