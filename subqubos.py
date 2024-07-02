import logging
import time
from collections import defaultdict
import random
from typing import Mapping, Hashable, Tuple
import dimod
import networkx as nx
import numpy as np
from dimod import SimulatedAnnealingSampler
from matplotlib import pyplot as plt
from numpy import floating, integer
from subqubo.QUBO import QUBO
from subqubo.subqubo_utils import subqubo_solve, sanitize_df
import pyomo.environ as pyo


def solve_model(qubo: QUBO, sense: pyo.kernel.objective) -> float:
    N = len(qubo.qubo_matrix)
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(N), within=pyo.Binary)

    model.obj = pyo.Objective(expr=sum(qubo.qubo_matrix[i, j] * model.x[i] * model.x[j]
                                       for i in range(N) for j in range(N)), sense=sense)
    solver = pyo.SolverFactory('cplex_direct')
    _ = solver.solve(model, tee=False)
    optimized_x = [pyo.value(model.x[i]) for i in range(N)]

    return sum(qubo.qubo_matrix[i, j] * optimized_x[i] * optimized_x[j] for i in range(N) for j in range(N))


def measure(variables: int, cut_dim: int, qubo: QUBO) -> None:
    min_sol, max_sol = solve_model(qubo, pyo.minimize), solve_model(qubo, pyo.maximize)
    start = time.time()
    subqubos = subqubo_solve(SimulatedAnnealingSampler(), qubo, dim=variables, cut_dim=cut_dim)
    end = time.time()
    log.info(f'Execution time for subqubo solver: {end - start:.2f}s')

    log.info(f'Ground truth solutions range from {np.round(min_sol, 5)} and {np.round(max_sol, 5)}')
    log.info(f'The best proposed solution has energy {min(sanitize_df(subqubos).energy)}')


def max_cut_problem() -> Tuple[nx.Graph, Mapping[tuple[Hashable, Hashable], float | floating | integer]]:
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5), (1, 6),
                          (2, 6), (4, 7), (1, 7), (2, 8), (3, 8), (1, 4), (2, 3)])

    qubo = defaultdict(int)
    for i, j in graph.edges:
        qubo[(i, i)] += -1
        qubo[(j, j)] += -1
        qubo[(i, j)] += 2
    return graph, qubo


def test_scale() -> None:
    test_set = [
        {'problem': max_cut_problem(), 'variables': 8, 'cut_dim': 2, 'name': 'max_cut'},
        {'variables': 8, 'cut_dim': 8, 'name': '8 vars, cut dim 8'},
        {'variables': 8, 'cut_dim': 4, 'name': '8 vars, cut dim 4'},
        {'variables': 8, 'cut_dim': 2, 'name': '8 vars, cut dim 2'},
        {'variables': 16, 'cut_dim': 16, 'name': '16 vars, cut dim 16'},
        {'variables': 16, 'cut_dim': 8, 'name': '16 vars, cut dim 8'},
        {'variables': 16, 'cut_dim': 4, 'name': '16 vars, cut dim 4'},
        {'variables': 16, 'cut_dim': 2, 'name': '16 vars, cut dim 2'},
        {'variables': 32, 'cut_dim': 32, 'name': '32 vars, cut dim 32'},
        {'variables': 32, 'cut_dim': 16, 'name': '32 vars, cut dim 16'},
        {'variables': 32, 'cut_dim': 8, 'name': '32 vars, cut dim 8'},
        {'variables': 32, 'cut_dim': 4, 'name': '32 vars, cut dim 4'},
        {'variables': 32, 'cut_dim': 2, 'name': '32 vars, cut dim 2'},
        {'variables': 64, 'cut_dim': 64, 'name': '64 vars, cut dim 64'},
        {'variables': 64, 'cut_dim': 32, 'name': '64 vars, cut dim 32'},
        {'variables': 64, 'cut_dim': 16, 'name': '64 vars, cut dim 16'},
        {'variables': 64, 'cut_dim': 8, 'name': '64 vars, cut dim 8'},
        {'variables': 64, 'cut_dim': 4, 'name': '64 vars, cut dim 4'},
        {'variables': 64, 'cut_dim': 2, 'name': '64 vars, cut dim 2'},
    ]

    for test_dict in test_set:
        log.info(f'Test {test_dict["name"]}'.upper())
        log.info(f'Variables: {test_dict["variables"]}'.upper())
        for j in range(10):
            log.info(f'Execution {j + 1}')
            if 'problem' in test_dict:
                qubo = QUBO(test_dict['problem'][1], cols_idx=[i + 1 for i in range(8)],
                            rows_idx=[i + 1 for i in range(8)])
                nx.draw(test_dict['problem'][0], with_labels=True, pos=nx.spectral_layout(test_dict['problem'][0]))
                plt.savefig("graph.png", format="PNG")
            else:
                num_interactions = random.randint(test_dict['variables'], test_dict['variables'] ** 2)
                log.info(f'Interactions: {num_interactions}'.upper())
                qubo = QUBO(dimod.generators.gnm_random_bqm(variables=test_dict['variables'],
                                                            num_interactions=num_interactions,
                                                            vartype=dimod.BINARY).to_qubo()[0],
                            [i for i in range(test_dict['variables'])],
                            [i for i in range(test_dict['variables'])])

            measure(test_dict['variables'], test_dict['cut_dim'], qubo)


def test_cut_dim() -> None:
    tests = [
        {'variables': 64, 'cut_dim': 64, 'name': '64 vars, cut dim 64'},
        {'variables': 64, 'cut_dim': 32, 'name': '64 vars, cut dim 32'},
        {'variables': 64, 'cut_dim': 16, 'name': '64 vars, cut dim 16'},
        {'variables': 64, 'cut_dim': 8, 'name': '64 vars, cut dim 8'},
        {'variables': 64, 'cut_dim': 4, 'name': '64 vars, cut dim 4'},
        {'variables': 64, 'cut_dim': 2, 'name': '64 vars, cut dim 2'}
    ]

    for test_dict in tests:
        log.info(f'Test {test_dict["name"]}'.upper())
        log.info(f'Variables: {test_dict["variables"]}'.upper())
        num_interactions = test_dict['variables']
        while num_interactions <= test_dict['variables'] ** 2:
            log.info(f'Interactions: {num_interactions}'.upper())
            qubo = QUBO(dimod.generators.gnm_random_bqm(variables=test_dict['variables'],
                                                        num_interactions=num_interactions,
                                                        vartype=dimod.BINARY).to_qubo()[0],
                        [i for i in range(test_dict['variables'])],
                        [i for i in range(test_dict['variables'])])
            measure(test_dict['variables'], test_dict['cut_dim'], qubo)
            num_interactions *= 2


if __name__ == '__main__':
    log = logging.getLogger('subqubo')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    file_handler = logging.FileHandler('subqubo.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info('SCALE TEST')
    test_scale()
    log.info('CUT TEST')
    test_cut_dim()
