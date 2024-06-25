import random
import dimod
import matplotlib.pyplot as plt
import pandas as pd
from subqubo.QUBO import QUBO
from subqubos import max_cut_problem


def plots1(problem: dict[str, int | float | str], direct_solutions: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.axhline(y=problem['sol'], color='r', linestyle='--', linewidth=1.5)
    plt.plot(range(len(direct_solutions)), direct_solutions['energy'], marker='o', linestyle='-', color='b')
    plt.title(problem['name'])
    plt.xlabel('Assignment')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.savefig('./img/subqubo/plot1/' + problem['name'] + '.png')


def plots2(problem: dict[str, int | float | str], direct_solutions: pd.DataFrame) -> None:
    energy_counts = direct_solutions['energy'].round(1).value_counts().reset_index()
    energy_counts.columns = ['energy', 'count']
    energy_counts = energy_counts.sort_values(by='energy')
    plt.figure(figsize=(10, 6))
    plt.axvline(x=problem['sol'], color='r', linestyle='--', linewidth=1.5)
    plt.plot(energy_counts['energy'], energy_counts['count'], marker='o', linestyle='-')
    plt.title(problem['name'])
    plt.xlabel('Valore di energy')
    plt.ylabel('Conteggio')
    plt.grid(True)
    plt.savefig('./img/subqubo/plot2/' + problem['name'] + '.png')


def main():
    problems = [
        {'problem': max_cut_problem()[1], 'variables': 8, 'cut_dim': 2, 'name': 'max_cut', 'sol': -10},
        {'variables': 8, 'cut_dim': 2, 'seed': 1997, 'name': '8 vars, cut dim 2', 'sol': 0},
        {'variables': 8, 'cut_dim': 4, 'seed': 1997, 'name': '8 vars, cut dim 4', 'sol': 0.95953},
        {'variables': 16, 'cut_dim': 8, 'seed': 1997, 'name': '16 vars, cut dim 8', 'sol': -3.83798},
        {'variables': 16, 'cut_dim': 4, 'seed': 1997, 'name': '16 vars, cut dim 4', 'sol': -19.00662},
    ]

    for problem in problems:
        if 'problem' in problem:
            qubo = QUBO(problem['problem'], cols_idx=[i + 1 for i in range(8)], rows_idx=[i + 1 for i in range(8)])
            del problem['problem']
        else:
            random.seed(problem['seed'])
            num_interactions = random.randint(problem['variables'], problem['variables'] ** 2)
            qubo = QUBO(dimod.generators.gnm_random_bqm(variables=problem['variables'],
                                                        num_interactions=num_interactions,
                                                        vartype=dimod.BINARY,
                                                        random_state=problem['seed']).to_qubo()[0],
                        [i for i in range(problem['variables'])],
                        [i for i in range(problem['variables'])])

        direct_solutions = (dimod.ExactSolver().sample_qubo(qubo.qubo_dict).to_pandas_dataframe()
                            .drop(columns=['num_occurrences']).drop_duplicates()
                            .sort_values(by='energy', ascending=True).reset_index(drop=True))

        plots1(problem, direct_solutions)
        plots2(problem, direct_solutions)


if __name__ == '__main__':
    main()
