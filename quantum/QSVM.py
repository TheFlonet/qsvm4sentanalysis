import itertools
from typing import Callable, List
import numpy as np
from dwave.preprocessing import Presolver
from sklearn.base import BaseEstimator, ClassifierMixin
import dimod
import dwave.system.samplers as dwavesampler


class QSVM(BaseEstimator, ClassifierMixin):
    """
    SVM dual problem with kernel trick and soft margin

    maximize_{alpha} -0.5 sum_{i=1}^N sum_{j=1}^N (alpha_i * alpha_j * y_i * y_j * K(x_i * x_j)) + sum_{i=1}^N alpha_i

    subject to:
        (1) C >= alpha_i >= 0 i in {1...N}
        (2) sum_{i=1}^N y_i * alpha_i = 0

    Where:
    - alpha_i are real number (optimization variables)
    - y_i are the labels (+/- 1)
    - K is the kernel matrix
    - x_i is the input vector
    - C is a bound on the error

    Since quantum solvers involve a minimisation problem, the objective function becomes

    minimize_{alpha}
        0.5 sum_{i=1}^N sum_{j=1}^N (
            alpha_i * alpha_j * y_i * y_j * K(x_i * x_j)
        ) - sum_{i=1}^N alpha_i

    In this case constraint (1) can be omitted because the value range is specified at the variable creation stage
    """

    def __init__(self, big_c: int, kernel: Callable[[np.ndarray, np.ndarray, float], np.ndarray]):
        self.big_c: int = big_c
        self.kernel: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = kernel
        self.support_vectors_labels: List[np.array] = []
        self.support_vectors_alphas: List[np.array] = []
        self.support_vectors_examples: List[np.array] = []
        self.b: List[float] = []
        self.ensemble_dim: int
        self.ensemble_w: List[float]

    def fit(self, examples: np.ndarray, labels: np.ndarray) -> None:
        n_samples, n_features = examples.shape
        N = range(n_samples)
        cqm = dimod.ConstrainedQuadraticModel()
        alphas = [dimod.Integer(label=f'alpha_{i}', lower_bound=0, upper_bound=self.big_c) for i in range(len(labels))]
        gamma = 1 / n_features
        kernel_matrix = np.array([[self.kernel(x1, x2, gamma) for x1 in examples] for x2 in examples])

        cqm.set_objective(0.5 * sum(labels[i] * alphas[i] * kernel_matrix[i, j] * labels[j] * alphas[j]
                                    for i, j in itertools.product(N, N)) - sum(alphas))
        cqm.add_constraint_from_comparison(sum(alpha * label for label, alpha in zip(labels, alphas)) == 0)
        presolve = Presolver(cqm)
        print('Is model pre-solvable?'.upper(), presolve.apply())
        reduced_cqm = presolve.detach_model()
        solver = dwavesampler.LeapHybridCQMSampler()
        reduced_sampleset = solver.sample_cqm(reduced_cqm, label='QSVM', time_limit=10)  # min time limit = 5 (default)
        sampleset = dimod.SampleSet.from_samples_cqm(presolve.restore_samples(reduced_sampleset.samples()), cqm)
        self.__extract_solution(examples, labels, kernel_matrix, sampleset)

    @staticmethod
    def __softmax(x: np.ndarray) -> np.ndarray:
        x = np.vectorize(round)(-x, 5)
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        softmax_x = exp_x / sum_exp_x
        return softmax_x

    def __extract_solution(self, examples: np.ndarray, labels: np.ndarray,
                           kernel_matrix: np.ndarray, sampleset: dimod.SampleSet) -> None:
        df = sampleset.to_pandas_dataframe()
        df = df.loc[(df['is_feasible']) & (df['is_satisfied'])]
        df = df.drop(['is_feasible', 'is_satisfied', 'num_occurrences'], axis=1)
        selected = df.loc[df['energy'] == min(df['energy'])]
        self.ensemble_w = self.__softmax(np.array(selected['energy']))
        selected = selected.drop('energy', axis=1)
        self.ensemble_dim = len(selected)
        for _, row in selected.iterrows():
            indices, alphas = [], []
            for i in range(len(row)):
                col = f'alpha_{i}'
                if 0 < row[col] < self.big_c:
                    indices.append(i)
                    alphas.append(int(row[col]))
            self.support_vectors_examples.append(examples[indices])
            self.support_vectors_alphas.append(np.array(alphas))
            self.support_vectors_labels.append(labels[indices])
            self.b.append(np.average([labels[original_i] - sum(alphas[real_j] * labels[original_j]
                                                               * kernel_matrix[original_i, original_j]
                                                               for real_j, original_j in enumerate(indices))
                                      for real_i, original_i in enumerate(indices)]))

    def predict(self, examples: np.ndarray) -> np.ndarray:
        if any(x is None for x in [self.support_vectors_examples, self.support_vectors_labels,
                                   self.support_vectors_alphas, self.b]):
            raise Exception('You need to fit before predicting.')
        predictions = np.ndarray(shape=(examples.shape[0], self.ensemble_dim))

        for i in range(examples.shape[0]):
            for j in range(self.ensemble_dim):
                predictions[i, j] = np.sign(sum(self.support_vectors_alphas[j][k] * self.support_vectors_labels[j][k]
                                                * self.kernel(self.support_vectors_examples[j][k],
                                                              examples[i], 1 / examples.shape[1])
                                                for k in range(len(self.support_vectors_alphas[j]))) + self.b[j])
        res = np.dot(predictions, self.ensemble_w)
        return np.sign(res).astype(int)
