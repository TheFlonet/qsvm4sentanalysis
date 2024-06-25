import logging
import time
from collections import defaultdict
import random
from typing import Mapping, Hashable, Tuple
import dimod
import networkx as nx
from dimod import SimulatedAnnealingSampler
from matplotlib import pyplot as plt
from numpy import floating, integer
from subqubo.QUBO import QUBO
from subqubo.subqubo_utils import subqubo_solve, compare_solutions


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


def tests() -> None:
    test_set = [
        {'problem': max_cut_problem(), 'variables': 8, 'cut_dim': 2, 'name': 'max_cut'},
        {'variables': 8, 'cut_dim': 2, 'seed': 1997, 'name': '8 vars, cut dim 2'},
        {'variables': 8, 'cut_dim': 4, 'seed': 1997, 'name': '8 vars, cut dim 4'},
        {'variables': 16, 'cut_dim': 8, 'seed': 1997, 'name': '16 vars, cut dim 8'},
        # {'variables': 16, 'cut_dim': 4, 'seed': 1997, 'name': '16 vars, cut dim 4'},
        # {'variables': 16, 'cut_dim': 2, 'seed': 1997, 'name': '16 vars, cut dim 2'},
    ]

    for idx, test_dict in enumerate(test_set):
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
                # random.seed(test_dict['seed'])
                num_interactions = random.randint(test_dict['variables'], test_dict['variables'] ** 2)
                log.info(f'Interactions: {num_interactions}'.upper())
                qubo = QUBO(dimod.generators.gnm_random_bqm(variables=test_dict['variables'],
                                                            num_interactions=num_interactions,
                                                            # random_state=test_dict['seed'],
                                                            vartype=dimod.BINARY).to_qubo()[0],
                            [i for i in range(test_dict['variables'])],
                            [i for i in range(test_dict['variables'])])

            direct_solutions = (dimod.ExactSolver().sample_qubo(qubo.qubo_dict).to_pandas_dataframe()
                                .drop(columns=['num_occurrences']).drop_duplicates()
                                .sort_values(by='energy', ascending=True))
            start = time.time()
            subqubos = subqubo_solve(SimulatedAnnealingSampler(), qubo,
                                     dim=test_dict['variables'], cut_dim=test_dict['cut_dim'])
            end = time.time()
            log.info(f'Execution time for subqubo solver: {end - start:.2f}s')
            compare_solutions(direct_solutions, subqubos)


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
    tests()
