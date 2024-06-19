import logging
from collections import defaultdict
from typing import Mapping, Hashable, Tuple
import networkx as nx
from dwave.samplers import SimulatedAnnealingSampler
from matplotlib import pyplot as plt
from numpy import floating, integer
from subqubo.subqubo_utils import subqubo_solve, compare_solutions


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
