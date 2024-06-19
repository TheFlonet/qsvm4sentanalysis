import logging
import os
from typing import List
import numpy as np
from dwave.preprocessing import Presolver
from sklearn.base import BaseEstimator, ClassifierMixin
import dimod
import dwave.system.samplers as dwavesampler
from util.kernel import rbf_kernel_pair
from util.model_generation import construct_svm_model
import math
import pandas as pd

log = logging.getLogger('qsvm')


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

    def __init__(self, big_c: int):
        self.big_c: int = big_c
        self.support_vectors_labels: List[np.array] = []
        self.support_vectors_alphas: List[np.array] = []
        self.support_vectors_examples: List[np.array] = []
        self.b: List[float] = []
        self.ensemble_dim: int

    def fit(self, examples: np.ndarray, labels: np.ndarray) -> None:
        model, kernel_matrix = construct_svm_model(examples, labels, self.big_c, round_to_int=True)
        model.write('qsvm.lp')
        cqm = dimod.lp.load('qsvm.lp')
        os.remove('qsvm.lp')
        presolve = Presolver(cqm)
        log.info(f'Is model pre-solvable? {presolve.apply()}'.upper())
        reduced_cqm = presolve.detach_model()
        log.info('Solving'.upper())
        solver = dwavesampler.LeapHybridCQMSampler()
        log.info(f'Min time required on QPU: {math.ceil(solver.min_time_limit(reduced_cqm))}s'.upper())
        reduced_sampleset = solver.sample_cqm(reduced_cqm, label='QSVM',
                                              time_limit=math.ceil(solver.min_time_limit(reduced_cqm)))
        sampleset = dimod.SampleSet.from_samples_cqm(presolve.restore_samples(reduced_sampleset.samples()), cqm)
        log.info('Extracting support vectors'.upper())
        self.__extract_solution(examples, labels, kernel_matrix, sampleset)

    def __extract_solution(self, examples: np.ndarray, labels: np.ndarray,
                           kernel_matrix: np.ndarray, sampleset: dimod.SampleSet) -> None:
        df = sampleset.to_pandas_dataframe()
        df = df.loc[(df['is_feasible']) & (df['is_satisfied'])]
        df = df.drop(['is_feasible', 'is_satisfied', 'num_occurrences'], axis=1)
        # selected = df.loc[df['energy'] == min(df['energy'])]
        selected = pd.DataFrame([df.loc[df['energy'].idxmin()]])
        selected = selected.drop('energy', axis=1)
        self.ensemble_dim = len(selected)
        for _, row in selected.iterrows():
            indices, alphas = [], []
            for i in range(len(row)):
                if 0 < row[f'x{i + 2}'] < self.big_c:
                    indices.append(i)
                    alphas.append(int(row[f'x{i + 2}']))
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
                                                * rbf_kernel_pair(self.support_vectors_examples[j][k],
                                                                  examples[i], 1 / examples.shape[1])
                                                for k in range(len(self.support_vectors_alphas[j]))) + self.b[j])
        return np.sign(predictions).astype(int)
