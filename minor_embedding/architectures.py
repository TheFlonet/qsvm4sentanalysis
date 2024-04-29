import itertools
import dimod
import numpy as np
from dwave.preprocessing import Presolver
from sklearn.model_selection import train_test_split
from dataset.dataset_creation import get_dummy_dataset


def svm_dummy_dataset():
    examples, labels = get_dummy_dataset()
    examples, _, labels, _ = train_test_split(examples, labels, test_size=0.4, random_state=7)

    def kernel(x1, x2, gamma):
        return np.exp(-np.linalg.norm(x1 - x2, ord=2) / (2 * (gamma ** 2)))

    n_samples, n_features = examples.shape
    N = range(n_samples)
    cqm = dimod.ConstrainedQuadraticModel()
    alphas = [dimod.Integer(label=f'alpha_{i}', lower_bound=0, upper_bound=20) for i in range(len(labels))]
    kernel_matrix = np.array([[kernel(x1, x2, 1 / n_features) for x1 in examples] for x2 in examples])

    cqm.set_objective(0.5 * sum(labels[i] * alphas[i] * kernel_matrix[i, j] * labels[j] * alphas[j]
                                for i, j in itertools.product(N, N)) - sum(alphas))
    cqm.add_constraint_from_comparison(sum(alpha * label for label, alpha in zip(labels, alphas)) == 0)
    presolve = Presolver(cqm)
    presolve.apply()
    bqm, _ = dimod.cqm_to_bqm(presolve.detach_model())
    return dimod.to_networkx_graph(bqm)
