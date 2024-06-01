import logging
from dotenv import load_dotenv
from dataset.dataset_creation import get_dummy_dataset
from sklearn.model_selection import train_test_split
from util.time_elapsed import eval_time
from util.visualization import *
from util.evaluation import evaluate
from sklearn.svm import SVC
from classical.CSVM import CSVM
from quantum.QSVM import QSVM


@eval_time
def main_c(examples_train: np.array, examples_test: np.array, labels_train: np.array, labels_test: np.array) -> None:
    svm_model = SVC(kernel='rbf', gamma='auto')
    log.info(f'Training with sklearn'.upper())
    svm_model.fit(examples_train, labels_train)
    plot_decision_boundary(svm_model, examples_train, labels_train, 'sklearn')
    predictions = svm_model.predict(examples_test)
    log.info(f'Testing with sklearn'.upper())
    evaluate(labels_test, predictions)


@eval_time
def main_opt(examples_train: np.array, examples_test: np.array, labels_train: np.array, labels_test: np.array) -> None:
    svm_model = CSVM(big_c=255)
    log.info(f'Training with cplex'.upper())
    svm_model.fit(examples_train, labels_train)
    plot_decision_boundary(svm_model, examples_train, labels_train, 'cplex')
    predictions = svm_model.predict(examples_test)
    log.info(f'Testing with cplex'.upper())
    evaluate(labels_test, predictions)


@eval_time
def main_q(examples_train: np.array, examples_test: np.array, labels_train: np.array, labels_test: np.array) -> None:
    svm_model = QSVM(big_c=255)
    log.info(f'Training with d-wave'.upper())
    svm_model.fit(examples_train, labels_train)
    plot_decision_boundary(svm_model, examples_train, labels_train, 'dwave')
    predictions = svm_model.predict(examples_test)
    log.info(f'Testing with d-wave'.upper())
    evaluate(labels_test, predictions)


def compare_svm() -> None:
    load_dotenv()
    log.info('Generating dataset'.upper())
    examples, labels = get_dummy_dataset()
    plot_dataset(examples, labels)
    ex_train, ex_test, l_train, l_test = train_test_split(examples, labels, test_size=0.4, random_state=7)

    main_c(ex_train, ex_test, l_train, l_test)
    main_opt(ex_train, ex_test, l_train, l_test)
    main_q(ex_train, ex_test, l_train, l_test)


if __name__ == '__main__':
    log = logging.getLogger('qsvm')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    compare_svm()
