import logging
import time
from collections import defaultdict
from typing import Mapping, Hashable, Tuple, Dict, List
import dimod
import networkx as nx
import numpy as np
import pandas as pd
from dimod import SimulatedAnnealingSampler
from matplotlib import pyplot as plt
from numpy import floating, integer
from subqubo.QSplitSampler import QSplitSampler
from subqubo.QUBO import QUBO


def sanitize_df(qubo: QUBO) -> pd.DataFrame:
    for idx, row in qubo.solutions.iterrows():
        x = row.drop('energy').values.astype(int)
        qubo.solutions.at[idx, 'energy'] = x @ qubo.qubo_matrix @ x.T

    return qubo.solutions


def measure(problem: Dict, interactions: int, trial: int, qubo: QUBO, res: Dict[str, List]) -> Dict[str, List]:
    direct_solutions = (dimod.RandomSampler().sample_qubo(qubo.qubo_dict, num_reads=100_000)
                        .to_pandas_dataframe().drop(columns=['num_occurrences']).drop_duplicates())
    start = time.time()
    sampler = QSplitSampler(SimulatedAnnealingSampler(), problem['cut_dim'])
    subqubos = sampler.run(qubo, problem['variables'])
    end = time.time()

    res['Name'].append(problem['name'])
    res['Trial'].append(trial)
    res['Variables'].append(problem['variables'])
    res['Cut dim'].append(problem['cut_dim'])
    res['Interactions'].append(interactions)
    res['Time (s)'].append(np.round(end - start, 5))
    res['Proposed solution'].append(np.round(min(sanitize_df(subqubos).energy), 5))
    res['Real sample min'].append(np.round(direct_solutions['energy'].min(), 5))
    res['Real sample 25'].append(np.round(direct_solutions.quantile(q=0.25)['energy'], 5))
    res['Real sample mean'].append(np.round(direct_solutions['energy'].mean(), 5))
    res['Real sample 75'].append(np.round(direct_solutions.quantile(q=0.75)['energy'], 5))
    res['Real sample max'].append(np.round(direct_solutions['energy'].max(), 5))

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
    test_set = [{'problem': max_cut_problem(), 'variables': 8, 'cut_dim': 2, 'name': 'Max cut'}]
    for variables in [8, 16, 32, 64, 128]:
        for cut_dim in [2, 4, 8, 16, 32, 64, 128]:
            if cut_dim <= variables:
                test_set.append({'variables': variables, 'cut_dim': cut_dim, 'name': f'V{variables}C{cut_dim}'})
    res = {
        'Name': [], 'Trial': [], 'Variables': [], 'Cut dim': [], 'Interactions': [], 'Time (s)': [],
        'Proposed solution': [], 'Real sample min': [], 'Real sample 25': [],
        'Real sample mean': [], 'Real sample 75': [], 'Real sample max': []
    }

    for test_dict in test_set:
        log.info(f'Test {test_dict["name"]}'.upper())
        for j in range(5):
            log.info(f'Execution {j + 1}')
            if 'problem' in test_dict:
                qubo = QUBO(test_dict['problem'][1], cols_idx=[i + 1 for i in range(8)],
                            rows_idx=[i + 1 for i in range(8)])
                num_interactions = 14
                nx.draw(test_dict['problem'][0], with_labels=True, pos=nx.spectral_layout(test_dict['problem'][0]))
                plt.savefig("graph.png", format="PNG")
                res = measure(test_dict, num_interactions, j, qubo, res)
            else:
                num_interactions = test_dict['variables']
                while num_interactions <= test_dict['variables'] ** 2:
                    qubo = QUBO(dimod.generators.gnm_random_bqm(variables=test_dict['variables'],
                                                                num_interactions=num_interactions,
                                                                vartype=dimod.BINARY).to_qubo()[0],
                                [i for i in range(test_dict['variables'])],
                                [i for i in range(test_dict['variables'])])
                    res = measure(test_dict, num_interactions, j, qubo, res)
                    num_interactions *= 4
    pd.DataFrame(res).to_csv("subqubo.csv", index=False)


if __name__ == '__main__':
    log = logging.getLogger('subqubo')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    test_scale()
