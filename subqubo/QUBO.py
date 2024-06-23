import numpy as np
from scipy.linalg import lu


class QUBO:
    def __init__(self, qubo_dict, cols_idx, rows_idx):
        self.qubo_dict = qubo_dict
        self.cols_idx = cols_idx
        self.rows_idx = rows_idx
        self.solutions = None
        self.qubo_matrix = None
        self.__from_dict_to_matrix((len(rows_idx), len(cols_idx)))

    def __is_upper_triangular(self):
        rows, cols = self.qubo_matrix.shape
        if rows != cols:
            return False
        for i in range(rows):
            for j in range(i):
                if self.qubo_matrix[i, j] != 0:
                    return False
        return True

    def __from_dict_to_matrix(self, dims):
        self.qubo_matrix = np.zeros(dims)
        for k, v in self.qubo_dict.items():
            row, col = k[0] - 1, k[1] - 1
            self.qubo_matrix[row % dims[0], col % dims[1]] = v
        if not self.__is_upper_triangular():
            self.qubo_matrix = lu(self.qubo_matrix, permute_l=True)[1]
        self.__from_matrix_to_dict()

    def __from_matrix_to_dict(self):
        self.qubo_dict = {}
        for i in range(self.qubo_matrix.shape[0]):
            for j in range(self.qubo_matrix.shape[1]):
                if self.qubo_matrix[i, j] != 0:
                    self.qubo_dict[(i + min(self.rows_idx), j + min(self.cols_idx))] = float(self.qubo_matrix[i, j])
