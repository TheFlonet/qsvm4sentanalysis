import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pyomo.environ as pyo
from util.kernel import rbf_kernel_pair
from util.model_generation import construct_svm_model

log = logging.getLogger('qsvm')


class CSVM(BaseEstimator, ClassifierMixin):
    """
    SVM dual problem with kernel trick and soft margin

    maximize_alpha
        -0.5 sum_{i=1}^N sum_{j=1}^N (
            alpha_i * alpha_j * y_i * y_j * K(x_i * x_j)
        ) + sum_{i=1}^N alpha_i
    subject to:
        (1) C >= alpha_i >= 0 i in {1...N}
        (2) sum_{i=1}^N y_i * alpha_i = 0

    Where:
    - alpha_i are real number (optimization variables)
    - y_i are the labels (+/- 1)
    - K is the kernel matrix
    - x_i is the input vector
    - C is a bound on the error
    """

    def __init__(self, big_c: int):
        self.big_c = big_c
        self.support_vectors_labels = None
        self.support_vectors_alphas = None
        self.support_vectors_examples = None
        self.b = None

    def fit(self, examples: np.ndarray, labels: np.ndarray) -> None:
        n_samples, _ = examples.shape
        N = range(n_samples)
        model, kernel_matrix = construct_svm_model(examples, labels, self.big_c)
        solver = pyo.SolverFactory('gurobi')
        log.info('Solving'.upper())
        results = solver.solve(model, tee=False)

        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            alphas = np.vectorize(round)(np.array([model.alpha[i].value for i in N]), ndigits=5)
            support_vector_indices = np.array([i for i, alpha in enumerate(alphas) if 0 < alpha < self.big_c])
            self.support_vectors_examples = examples[support_vector_indices]
            self.support_vectors_alphas = alphas[support_vector_indices]
            self.support_vectors_labels = labels[support_vector_indices]
            self.b = np.average([labels[i] - sum(alphas[j] * labels[j] * kernel_matrix[i, j]
                                                 for j in support_vector_indices) for i in support_vector_indices])
        else:
            raise Exception('Optimal solution was not found.')

    def predict(self, examples: np.ndarray) -> np.ndarray:
        if any(x is None for x in [self.support_vectors_examples, self.support_vectors_labels,
                                   self.support_vectors_alphas, self.b]):
            raise Exception('You need to fit before predicting.')
        gamma = 1 / examples.shape[1]
        return np.array([np.sign(sum(self.support_vectors_alphas[i] * self.support_vectors_labels[i]
                                     * rbf_kernel_pair(self.support_vectors_examples[i], example, gamma)
                                     for i in range(len(self.support_vectors_alphas))) + self.b)
                         for example in examples])
