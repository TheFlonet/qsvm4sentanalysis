import logging
from collections import defaultdict
from typing import Mapping, Hashable, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler
from matplotlib import pyplot as plt
from numpy import floating, integer
from subqubo.subqubo_utils import subqubo_solve


def generate_max_cut_problem() -> Tuple[nx.Graph, Mapping[tuple[Hashable, Hashable], float | floating | integer]]:
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5), (1, 6),
                          (2, 6), (4, 7), (1, 7), (2, 8), (3, 8), (1, 4), (2, 3)])

    qubo = defaultdict(int)
    for i, j in graph.edges:
        qubo[(i, i)] += -1
        qubo[(j, j)] += -1
        qubo[(i, j)] += 2
    return graph, qubo


def check_dataframe_consistency(ground_truth: pd.DataFrame, sol: pd.DataFrame,
                                qubo_matrix: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for idx, row in ground_truth.iterrows():
        x = row.drop('energy').values.astype(int)
        energy = x @ qubo_matrix @ x.T
        if energy != ground_truth.at[idx, 'energy']:
            log.warning('Incorrect energy value in ground truth')
            log.warning(f'Expected: {energy}, found: {ground_truth.at[idx, "energy"]}. Overriding the value')
            ground_truth.at[idx, 'energy'] = energy

    for idx, row in sol.iterrows():
        x = row.drop('energy').values.astype(int)
        energy = x @ qubo_matrix @ x.T
        if energy != sol.at[idx, 'energy']:
            log.warning('Incorrect energy value in proposed solution')
            log.warning(f'Expected: {energy}, found: {sol.at[idx, "energy"]}. Overriding the value')
            sol.at[idx, 'energy'] = energy

    return ground_truth, sol


def compare_solutions(ground_truth: pd.DataFrame, sol: pd.DataFrame,
                      qubo: Mapping[tuple[Hashable, Hashable], float | floating | integer], dim: int) -> None:
    qubo_matrix = np.zeros((dim, dim))
    for k, v in qubo.items():
        qubo_matrix[(k[0] - 1) % dim, (k[1] - 1) % dim] = v
    ground_truth, sol = check_dataframe_consistency(ground_truth, sol, qubo_matrix)

    log.info(f'The best ground truth solution has energy {min(ground_truth.energy)}')
    log.info(f'The best proposed solution has energy {min(sol.energy)}')
    if min(ground_truth.energy) == 0:
        gap = ((min(sol.energy) + 1 - min(ground_truth.energy) + 1) / abs(min(ground_truth.energy) + 1)) * 100
    else:
        gap = ((min(sol.energy) - min(ground_truth.energy)) / abs(min(ground_truth.energy))) * 100
    log.info(f'Relativa gap: {gap:.2f}%')


def main() -> None:
    graph, qubo = generate_max_cut_problem()

    nx.draw(graph, with_labels=True, pos=nx.spectral_layout(graph))
    plt.savefig("graph.png", format="PNG")

    sampler = SimulatedAnnealingSampler()
    direct_solutions = (sampler.sample_qubo(qubo, num_reads=10)
                        .to_pandas_dataframe()
                        .drop(columns=['num_occurrences'])
                        .drop_duplicates()
                        .sort_values(by='energy', ascending=True))
    problem_dim = len({i for k in qubo.keys() for i in k})

    subqubos_solutions = subqubo_solve(sampler, qubo, problem_dim)
    compare_solutions(direct_solutions, subqubos_solutions, qubo, problem_dim)


if __name__ == '__main__':
    log = logging.getLogger('subqubo')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    main()
