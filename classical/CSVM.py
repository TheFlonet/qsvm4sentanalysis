import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pyomo.environ as pyo


class CSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, kernel='rbf', gamma=0.1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_alphas_ = None
        self.intercept_ = None

    def fit(self, examples, labels):
        n_samples, n_features = examples.shape

        model = pyo.ConcreteModel()
        model.alpha = pyo.Var(range(n_samples), domain=pyo.NonNegativeReals)

        def objective_rule(working_model):
            objective = sum(working_model.alpha[i] for i in range(n_samples))
            kernel_matrix = self.compute_kernel_matrix(examples)
            for i in range(n_samples):
                for j in range(n_samples):
                    objective -= (0.5 * working_model.alpha[i] * working_model.alpha[j] *
                                  labels[i] * labels[j] * kernel_matrix[i, j])
            return objective

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # Constraints: sum(alpha_i * y_i) = 0 e 0 <= alpha_i <= C
        def equality_constraint_rule(working_model):
            return sum(working_model.alpha[i] * labels[i] for i in range(n_samples)) == 0

        def inequality_constraint_rule(working_model, i):
            return working_model.alpha[i] <= self.C

        model.equality_constraint = pyo.Constraint(rule=equality_constraint_rule)
        model.inequality_constraints = pyo.Constraint(range(n_samples), rule=inequality_constraint_rule)

        solver = pyo.SolverFactory('gurobi')
        _ = solver.solve(model, tee=False)

        alphas = np.array([model.alpha[i].value for i in range(n_samples)])
        self.support_vectors_ = examples[alphas > 0]
        self.support_vector_labels_ = labels[alphas > 0]
        self.support_vector_alphas_ = alphas[alphas > 0]
        self.intercept_ = self.compute_intercept(examples, labels, alphas)

        return self

    def predict(self, examples):
        if self.support_vectors_ is None or self.support_vector_labels_ is None or self.support_vector_alphas_ is None:
            raise Exception("Model not trained yet")

        kernel_values = self.compute_kernel_matrix(examples, self.support_vectors_)
        decision_function = np.dot(kernel_values, self.support_vector_alphas_ * self.support_vector_labels_)
        decision_function += self.intercept_
        predictions = np.sign(decision_function)

        return predictions

    def compute_kernel_matrix(self, x1, x2=None):
        n_samples = x1.shape[0]
        if x2 is None:
            x2 = x1
        kernel_matrix = np.zeros((n_samples, n_samples))
        if self.kernel == 'rbf':
            for i in range(n_samples):
                for j in range(n_samples):
                    kernel_matrix[i, j] = np.exp(-self.gamma * np.linalg.norm(x1[i] - x2[j]) ** 2)
        return kernel_matrix

    def compute_intercept(self, examples, labels, alphas):
        index = -1
        for i in range(len(alphas)):
            if 0 < alphas[i] < self.C:
                index = i
                break
        if index == -1:
            raise ValueError("Alpha is out of range")
        intercept = labels[index] - np.dot(self.compute_kernel_matrix(examples[index], examples), alphas * labels)
        return intercept
