import pandas as pd
import time
from typing import Dict
from minor_embedding.architectures import scale_svm
import networkx as nx
from minor_embedding.minors import generate_pegasus
from minor_embedding.search_equals import generate_embedding_graph


def compute_data_entry(pegasus: nx.Graph, num_examples: int, prime: int, ub: int,
                       presolve: bool) -> Dict[str, int | bool]:
    svm, presolve_apply = scale_svm(num_examples=num_examples, alpha_ub=ub, presolve=presolve)
    st = time.time()
    g = generate_embedding_graph(svm, pegasus, prime)
    et = time.time()

    return {'seed': prime, 'num_examples': num_examples, 'ub': ub, 'is_valid': g is not None,
            'presolve': presolve, 'pre_solvable': presolve_apply, 'nodes': -1 if g is None else g.number_of_nodes(),
            'time': round(et - st, 5)}


def problem_scalability() -> None:
    pegasus = generate_pegasus()
    with open('../util/1000.prime') as f:
        primes = [int(x) for x in f.readline().split(', ')][:100]
    ub_list = [2 ** x - 1 for x in range(2, 9)]

    i = 0
    for prime in primes:
        df = pd.DataFrame(columns=['seed', 'num_examples', 'ub', 'is_valid', 'presolve',
                                   'pre_solvable', 'nodes', 'time'])
        valid = True
        num_examples = 2
        print('Computed for:', end=' ')
        while valid:
            print(num_examples, end=', ')
            for ub in ub_list:
                df.loc[i] = compute_data_entry(pegasus, num_examples, prime, ub, True)
                i += 1
                df.loc[i] = compute_data_entry(pegasus, num_examples, prime, ub, False)
                i += 1
                valid = df.loc[i - 1].is_valid or df.loc[i - 2].is_valid
            num_examples += 2

        print('\nSaving for prime', prime)
        df.to_csv(f'outputs/scale_{str(prime).zfill(4)}.csv', index=False)


if __name__ == '__main__':
    problem_scalability()
