import itertools
import dimod
import networkx as nx
import numpy as np
from dwave.preprocessing import Presolver
from sklearn.model_selection import train_test_split
from dataset.dataset_creation import get_dummy_dataset


def svm_dummy_dataset() -> nx.Graph:
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


def toy_svm(alpha_ub: int = 10, presolve: bool = False) -> nx.Graph:
    examples = np.array([[1, 0], [0, 1]])
    labels = np.array([-1, 1])
    cqm = dimod.ConstrainedQuadraticModel()
    alphas = [dimod.Integer(label=f'alpha_{i}', lower_bound=0, upper_bound=alpha_ub) for i in range(len(labels))]
    cqm.set_objective(0.5 * (alphas[0] * alphas[0] * labels[0] * labels[0] * examples[0] @ examples[0]
                             + alphas[0] * alphas[1] * labels[0] * labels[1] * examples[0] @ examples[1]
                             + alphas[1] * alphas[0] * labels[1] * labels[0] * examples[1] @ examples[0]
                             + alphas[1] * alphas[1] * labels[1] * labels[1] * examples[1] @ examples[1])
                      + sum(alphas))
    cqm.add_constraint_from_comparison(sum(alpha * label for label, alpha in zip(labels, alphas)) == 0)

    if presolve:
        cqm = Presolver(cqm)  # NOTE: presolve increase nodes number
        print('Is pre-solvable?', cqm.apply())
        cqm = cqm.detach_model()

    bqm, _ = dimod.cqm_to_bqm(cqm)
    return dimod.to_networkx_graph(bqm)
