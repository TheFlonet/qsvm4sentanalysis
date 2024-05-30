from typing import List, Mapping, Hashable
from dimod import SampleSet
from dwave.samplers import SimulatedAnnealingSampler
from numpy import integer, floating


def split_problem(qubo: Mapping[tuple[Hashable, Hashable], float | floating | integer],
                  halt_dim: int) -> List[Mapping[tuple[Hashable, Hashable], float | floating | integer]]:
    pass


def aggregate_solutions(solutions: List[SampleSet]) -> SampleSet:
    pass


def subqubo_solve(sampler: SimulatedAnnealingSampler,
                  qubo: Mapping[tuple[Hashable, Hashable], float | floating | integer]) -> SampleSet:
    qubos = split_problem(qubo, 2)
    solutions = []

    for q in qubos:
        solutions.append(sampler.sample_qubo(q))

    return aggregate_solutions(solutions)
