import dimod
import networkx as nx
from dimod import SampleSet
from dwave.samplers import SimulatedAnnealingSampler
from subqubo.subqubo_utils import subqubo_solve


def generate_max_cut_problem() -> nx.Graph:
    # Generate a graph that has at least 2 equivalent solutions to max-cut problem
    # The problem will generate a qubo matrix 2^n X 2^n
    pass


def compare_solutions(ground_truth: SampleSet, sol: SampleSet):
    pass


def main() -> None:
    graph = generate_max_cut_problem()
    bqm = dimod.from_networkx_graph(graph)

    sampler = SimulatedAnnealingSampler()
    direct_solutions = sampler.sample(bqm)
    subqubos_solutions = subqubo_solve(sampler, bqm.to_qubo())

    compare_solutions(direct_solutions, subqubos_solutions)


if __name__ == '__main__':
    main()
