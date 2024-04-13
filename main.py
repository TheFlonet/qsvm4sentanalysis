from dataset.dataset_creation import get_dummy_dataset
from sklearn.model_selection import train_test_split
from util.visualization import *
from util.evaluation import evaluate
from sklearn.svm import SVC
from classical.CSVM import CSVM
from quantum.QSVM import QSVM


def main_classic(examples_train, examples_test, labels_train, labels_test):
    svm_model = SVC(kernel='rbf', C=20, gamma='auto')
    print(f'Training with sklearn'.upper())
    svm_model.fit(examples_train, labels_train)
    plot_decision_boundary(svm_model, examples_train, labels_train)
    predictions = svm_model.predict(examples_test)
    print(f'Testing with sklearn'.upper())
    evaluate(labels_test, predictions)
    print('-' * 50)


def main_opt(examples_train, examples_test, labels_train, labels_test):
    gamma = 0.5
    svm_model = CSVM(big_c=20, kernel=lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2, ord=2) / (2 * (gamma ** 2))))
    print(f'Training with gurobi'.upper())
    svm_model.fit(examples_train, labels_train)
    plot_decision_boundary(svm_model, examples_train, labels_train)
    predictions = svm_model.predict(examples_test)
    print(f'Testing with gurobi'.upper())
    evaluate(labels_test, predictions)
    print('-' * 50)


def main_quantum(examples_train, examples_test, labels_train, labels_test):
    gamma = 0.5
    svm_model = QSVM(big_c=20, kernel=lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2, ord=2) / (2 * (gamma ** 2))))
    print(f'Training with d-wave'.upper())
    svm_model.fit(examples_train, labels_train)
    plot_decision_boundary(svm_model, examples_train, labels_train)
    predictions = svm_model.predict(examples_test)
    print(f'Testing with d-wave'.upper())
    evaluate(labels_test, predictions)
    print('-' * 50)


def main():
    print('Generating dataset'.upper())
    examples, labels = get_dummy_dataset()
    plot_dataset(examples, labels)
    ex_train, ex_test, l_train, l_test = train_test_split(examples, labels, test_size=0.4, random_state=7)
    main_classic(ex_train, ex_test, l_train, l_test)
    main_opt(ex_train, ex_test, l_train, l_test)
    # main_quantum(ex_train, ex_test, l_train, l_test)


if __name__ == '__main__':
    main()
