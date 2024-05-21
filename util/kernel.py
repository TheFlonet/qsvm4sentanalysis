import numpy as np


def rbf_kernel_matrix(vector: np.array, gamma: float) -> np.array:
    vector_norm = np.sum(vector ** 2, axis=1)
    sq_dists = vector_norm[:, np.newaxis] + vector_norm[np.newaxis, :] - 2 * np.dot(vector, vector.T)
    return np.exp(-gamma * sq_dists)


def rbf_kernel_pair(x1: np.array, x2: np.array, gamma: float) -> np.array:
    return np.exp(-gamma * (np.linalg.norm(x1 - x2, ord=2) ** 2))
