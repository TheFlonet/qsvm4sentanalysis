import logging
import os
from typing import Tuple

import dimod
from dimod import ConstrainedQuadraticModel

from util.kernel import rbf_kernel_matrix
import pyomo.environ as pyo
import numpy as np

log = logging.getLogger('qsvm')


def construct_svm_model(examples: np.ndarray, labels: np.ndarray, big_c: int,
                        round_to_int: bool = False) -> Tuple[pyo.ConcreteModel, np.array]:
    n_samples, n_features = examples.shape
    N = range(n_samples)
    log.info('Creating model'.upper())
    model = pyo.ConcreteModel()
    model.alpha = pyo.Var(N, domain=pyo.NonNegativeIntegers if round_to_int else pyo.NonNegativeReals,
                          bounds=(0, big_c))
    gamma = 1 / examples.shape[1]
    kernel_matrix = rbf_kernel_matrix(examples, gamma)
    model.objective = pyo.Objective(rule=lambda w_model: (
            0.5 * ((labels * model.alpha) @ kernel_matrix @ (labels * model.alpha).T) - sum(w_model.alpha[i] for i in N)
    ), sense=pyo.minimize)
    model.constraint1 = pyo.Constraint(rule=lambda w_model: sum(np.multiply(w_model.alpha, labels)) == 0)
    return model, kernel_matrix


def load_svm_model(path: str) -> Tuple[ConstrainedQuadraticModel, np.array]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Path {path} not found')

    kernel_file = os.path.join(path, 'svm_kernel.npy')
    model_int_lp_file = os.path.join(path, 'svm_model_int.lp')

    if not os.path.isfile(kernel_file):
        raise FileNotFoundError(f'Missing kernel matrix file in {kernel_file}')

    if not os.path.isfile(model_int_lp_file):
        raise FileNotFoundError(f'Missing model file in {model_int_lp_file}')

    return dimod.lp.load(model_int_lp_file), np.load(kernel_file)
