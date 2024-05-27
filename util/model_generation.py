import logging
from typing import Tuple
from util.kernel import rbf_kernel_matrix
import pyomo.environ as pyo
import numpy as np

log = logging.getLogger('qsvm')


def construct_svm_model(examples: np.ndarray, labels: np.ndarray, big_c: int) -> Tuple[pyo.ConcreteModel, np.array]:
    n_samples, n_features = examples.shape
    N = range(n_samples)
    log.info('Creating model'.upper())
    model = pyo.ConcreteModel()
    model.alpha = pyo.Var(N, domain=pyo.NonNegativeReals, bounds=(0, big_c))
    gamma = 1 / examples.shape[1]
    kernel_matrix = rbf_kernel_matrix(examples, gamma)
    model.objective = pyo.Objective(rule=lambda w_model: (
            sum(w_model.alpha[i] for i in N) -
            0.5 * ((labels * model.alpha) @ kernel_matrix @ (labels * model.alpha).T)
    ), sense=pyo.maximize)
    model.constraint1 = pyo.Constraint(rule=lambda w_model: sum(np.multiply(w_model.alpha, labels)) == 0)
    return model, kernel_matrix
