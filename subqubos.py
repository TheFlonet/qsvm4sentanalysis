import logging
import time
from collections import defaultdict
from typing import Mapping, Hashable, Tuple, Dict, List
import dimod
import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from numpy import floating, integer
from subqubo.QSplitSampler import QSplitSampler
from subqubo.QUBO import QUBO
import pyomo.environ as pyo
from dwave.system import EmbeddingComposite, DWaveSampler


def sanitize_df(qubo: QUBO) -> pd.DataFrame:
    for idx, row in qubo.solutions.iterrows():
        x = row.drop('energy').values.astype(int)
        qubo.solutions.at[idx, 'energy'] = x @ qubo.qubo_matrix @ x.T

    return qubo.solutions


def get_direct_sol(qubo: QUBO) -> pd.DataFrame:
    return (EmbeddingComposite(DWaveSampler()).sample_qubo(qubo.qubo_dict, num_reads=10)
            .to_pandas_dataframe().drop(columns=['num_occurrences']).drop_duplicates())


def get_random_sol(qubo: QUBO) -> pd.DataFrame:
    return (dimod.RandomSampler()
            .sample_qubo(qubo.qubo_dict, num_reads=min(500_000, 2 ** len(qubo.rows_idx)))
            .to_pandas_dataframe().drop(columns=['num_occurrences']).drop_duplicates())


def get_sol_range(qubo: QUBO) -> Tuple[float, float]:
    n = len(qubo.rows_idx)
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(n), domain=pyo.Binary)

    def objective_rule(w_model):
        return sum(qubo.qubo_matrix[i][j] * model.x[i] * model.x[j] for i in range(n) for j in range(n))

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize, name='obj')
    solver = pyo.SolverFactory('cplex_direct')
    solver.solve(model)
    min_sol = model.obj()
    model.del_component('obj')
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    solver = pyo.SolverFactory('cplex_direct')
    solver.solve(model)
    max_sol = model.obj()
    return min_sol, max_sol


def normalize_sol(solution: float, sol_range: Tuple[float, float]) -> float:
    return (solution - sol_range[0]) / (sol_range[1] - sol_range[0])


def measure(problem: Dict, qubo: QUBO, res: Dict[str, List]) -> Dict[str, List]:
    res['Name'].append(problem['name'])
    res['Variables'].append(problem['variables'])
    res['Cut dim'].append(problem['cut_dim'])
    sol_range = get_sol_range(qubo)

    start_time = time.time()
    direct_solutions = get_direct_sol(qubo)
    end_time = time.time()
    res['Direct time (s)'].append(np.round(end_time - start_time, 5))
    res['Direct sol'].append(np.round(normalize_sol(direct_solutions['energy'].min(), sol_range), 5))

    start_time = time.time()
    random_solutions = get_random_sol(qubo)
    end_time = time.time()
    res['Random time (s)'].append(np.round(end_time - start_time, 5))
    res['Random min sol'].append(np.round(normalize_sol(random_solutions['energy'].min(), sol_range), 5))
    res['Random avg sol'].append(np.round(normalize_sol(random_solutions['energy'].mean(), sol_range), 5))

    start = time.time()
    subqubos, qpu_time = QSplitSampler(EmbeddingComposite(DWaveSampler()),
                                       problem['cut_dim']).run(qubo, problem['variables'])
    end = time.time()
    res['QSplit CPU+Network time (s)'].append(np.round(end - start, 5))
    res['QSplit QPU time (s)'].append(np.round(qpu_time, 5))
    res['QSplit sol'].append(np.round(normalize_sol(sanitize_df(subqubos)['energy'].min(), sol_range), 5))

    return res


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
    load_dotenv()
    test_set = [{'problem': max_cut_problem(), 'variables': 8, 'cut_dim': 2, 'name': 'Max cut'}] + [
        {'variables': variables, 'cut_dim': cut_dim, 'name': f'V{variables}C{cut_dim}'}
        for variables in [8, 16, 32, 64, 128]
        for cut_dim in [2 ** i for i in range(1, min(variables.bit_length() - 1, 6))]
    ]
    res = {
        'Name': [], 'Variables': [], 'Cut dim': [],
        'QSplit CPU+Network time (s)': [], 'QSplit QPU time (s)': [], 'QSplit sol': [],
        'Direct time (s)': [], 'Direct sol': [],
        'Random time (s)': [], 'Random min sol': [], 'Random avg sol': []
    }

    for test_dict in test_set:
        log.info(f'Test {test_dict["name"]}'.upper())
        if 'problem' in test_dict:
            qubo = QUBO(test_dict['problem'][1], cols_idx=[i + 1 for i in range(8)],
                        rows_idx=[i + 1 for i in range(8)])
            nx.draw(test_dict['problem'][0], with_labels=True, pos=nx.spectral_layout(test_dict['problem'][0]))
            plt.savefig('graph.png', format='PNG')
            res = measure(test_dict, qubo, res)
        else:
            qubo = QUBO(dimod.generators.gnm_random_bqm(variables=test_dict['variables'],
                                                        num_interactions=test_dict['variables'] ** 2,
                                                        vartype=dimod.BINARY).to_qubo()[0],
                        [i for i in range(test_dict['variables'])],
                        [i for i in range(test_dict['variables'])])
            res = measure(test_dict, qubo, res)
        pd.DataFrame(res).to_csv('subqubo.csv', index=False)


if __name__ == '__main__':
    log = logging.getLogger('subqubo')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    test_scale()
