import itertools
import logging
from collections import defaultdict
from typing import List, Tuple, Any
import networkx as nx
import numpy as np
import pandas as pd
from dimod import SampleSet
from dimod.views.samples import SampleView
from dwave.samplers import SimulatedAnnealingSampler
from subqubo.QUBO import QUBO

log = logging.getLogger('subqubo')


def __split_problem(qubo: QUBO, dim: int) -> Tuple[QUBO, QUBO, QUBO]:
    """
        Returns 3 sub-problems in qubo form.
        The 3 sub-problems correspond to the matrices obtained by dividing the qubo matrix of the original problem
        in half both horizontally and vertically.
        The sub-problem for the sub-matrix in the bottom left corner is not given as this is always empty.
        The order of the results is:
        - Upper left sub-matrix,
        - Upper right sub-matrix,
        - Lower right sub-matrix.

        All sub-problems are converted to obtain an upper triangular matrix.
    """
    split_u_l = defaultdict(int)
    split_u_r = defaultdict(int)
    split_d_r = defaultdict(int)
    split_idx = dim // 2

    for k, v in qubo.qubo_dict.items():
        row, col = k[0] - 1, k[1] - 1
        if row < split_idx and col < split_idx:
            split_u_l[k] = v
        elif row < split_idx <= col:
            split_u_r[k] = v
        elif row >= split_idx and col >= split_idx:
            split_d_r[k] = v
        else:
            raise ValueError('All values in the lower left matrix should be 0, so not present in the qubo dictionary')

    res = (QUBO(split_u_l, cols_idx=[i + 1 for i in range(split_idx)],
                rows_idx=[i + 1 for i in range(split_idx)]),
           QUBO(split_u_r, cols_idx=[i + 1 + split_idx for i in range(split_idx)],
                rows_idx=[i + 1 for i in range(split_idx)]),
           QUBO(split_d_r, cols_idx=[i + 1 + split_idx for i in range(split_idx)],
                rows_idx=[i + 1 + split_idx for i in range(split_idx)]))

    return res


def __nan_hamming_distance(a: np.ndarray, b: np.ndarray) -> float | Any:
    mask = ~np.isnan(a) & ~np.isnan(b)
    if np.sum(mask) == 0:
        return np.inf
    return np.sum(a[mask] != b[mask]) / np.sum(mask)


def __combine_rows(row1: pd.Series, row2: pd.Series) -> List[float | Any]:
    combined_row = []
    is_row_consistent = True
    for col in row1.index:
        val1, val2 = row1[col], row2[col]
        if col == 'energy':
            if is_row_consistent:
                combined_row.append(val1 + val2)
            else:
                combined_row.append(np.nan)
        else:
            if pd.isna(val2):
                combined_row.append(val1)
            elif val1 == val2:
                combined_row.append(val1)
            else:
                combined_row.append(np.nan)
                is_row_consistent = is_row_consistent and False
    return combined_row


def __combine_ul_lr(ul: QUBO, lr: QUBO) -> pd.DataFrame:
    all_indices = sorted(list(set(ul.rows_idx).union(lr.cols_idx)))
    ul.solutions['tmp'] = 1
    lr.solutions['tmp'] = 1
    merge = pd.merge(ul.solutions, lr.solutions, on='tmp')
    merge['energy'] = merge['energy_x'] + merge['energy_y']
    merge = merge.drop(['energy_x', 'energy_y', 'tmp'], axis=1)
    ul.solutions.drop('tmp', axis=1, inplace=True)
    lr.solutions.drop('tmp', axis=1, inplace=True)
    merge = __fill_with_nan(pd.DataFrame(columns=all_indices + ['energy']), merge)

    return __expand_df(merge, keep_energy=True)


def __fill_with_nan(schema: pd.DataFrame, df_to_fill: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(schema.columns) - set(df_to_fill.columns)
    for col in missing_columns:
        df_to_fill[col] = np.nan
    return df_to_fill[schema.columns]


def __get_closest_assignments(starting_sols: pd.DataFrame, ur_qubo_filled: pd.DataFrame) -> pd.DataFrame:
    closest_rows = []
    for i, row in starting_sols.iterrows():
        distances = []
        for j, sol_row in ur_qubo_filled.iterrows():
            distance = __nan_hamming_distance(row.values, sol_row.values)
            distances.append(distance)
        closest_idx = np.argmin(distances)
        closest_rows.append(ur_qubo_filled.iloc[closest_idx])
    return pd.DataFrame(closest_rows).reset_index(drop=True)


def __expand_df(df: pd.DataFrame, keep_energy: bool) -> pd.DataFrame:
    complete_rows = df.dropna().copy()
    incomplete_rows = df[df.isna().any(axis=1)].copy()
    result_rows = complete_rows.to_dict('records')

    for idx, row in incomplete_rows.iterrows():
        nan_columns = row.index[row.isna()]
        non_nan_part = row.drop(nan_columns)
        combinations = list(itertools.product([0, 1], repeat=len(nan_columns)))

        for combination in combinations:
            new_values = dict(zip(nan_columns, combination))
            new_row = pd.Series({**non_nan_part.to_dict(), **new_values})
            new_row['energy'] = df.at[idx, 'energy'] if keep_energy else np.nan
            result_rows.append(new_row.to_dict())

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.drop_duplicates(subset=result_df.columns[:-1]).reset_index(drop=True)

    return result_df


def __brute_force(df: pd.DataFrame, qubo_matrix: np.ndarray) -> Tuple[pd.DataFrame, int]:
    result_df = __expand_df(df, keep_energy=False)

    trials = 0
    for idx, row in result_df.iterrows():
        if pd.isna(row['energy']):
            trials += 1
            x = row.drop('energy').values.astype(int)
            result_df.at[idx, 'energy'] = x @ qubo_matrix @ x.T

    return result_df.reset_index(drop=True), trials


def __aggregate_solutions(solutions: List[QUBO], qubo: QUBO) -> QUBO:
    # Aggregate upper-left qubo with lower-right
    starting_sols = __combine_ul_lr(solutions[0], solutions[2])
    # Set missing columns in upper-right qubo to NaN
    ur_qubo_filled = __fill_with_nan(starting_sols, solutions[1].solutions)
    # Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
    closest_df = __get_closest_assignments(starting_sols, ur_qubo_filled)

    # Combine
    combined_df = pd.DataFrame([__combine_rows(row1, row2) for (_, row1), (_, row2) in
                                zip(starting_sols.iterrows(), closest_df.iterrows())],
                               columns=starting_sols.columns)

    # Brute force resolution
    res, trials = __brute_force(combined_df, qubo.qubo_matrix)
    log.info(f'Dimension {qubo.qubo_matrix.shape[0]}, merged successfully.')
    log.info(f'    Conflicts resolved with classical resolutions: {trials}')
    qubo.solutions = res

    return qubo


def subqubo_solve(sampler: SimulatedAnnealingSampler, qubo: QUBO, dim: int, cut_dim: int) -> QUBO:
    if dim <= cut_dim:
        if len(qubo.qubo_dict) == 0:
            all_indices = sorted(list(set(qubo.rows_idx).union(qubo.cols_idx)))
            combinations = list(itertools.product([0, 1], repeat=len(all_indices)))
            binary_vectors_as_lists = [list(vec) for vec in combinations]
            data = [vec + [0] for vec in binary_vectors_as_lists]
            column_names = all_indices + ['energy']
            qubo.solutions = pd.DataFrame(data, columns=column_names)
        else:
            res = (sampler.sample_qubo(qubo.qubo_dict, num_reads=10)
                   .to_pandas_dataframe()
                   .drop(columns=['num_occurrences'])
                   .drop_duplicates()
                   .sort_values(by='energy', ascending=True))
            qubo.solutions = res[res['energy'] == min(res['energy'])]
        return qubo
    return __aggregate_solutions([subqubo_solve(sampler, q, dim // 2, cut_dim) for q in __split_problem(qubo, dim)],
                                 qubo)


def print_cut_from_sampleset(sampleset: SampleSet) -> None:
    """
    Adapted from the example available on the dwave GitHub profile:

    Available here: https://github.com/dwave-examples/maximum-cut/blob/master/maximum_cut.py
    """
    print('-' * 60)
    print(f'{"Set 0":>15s}{"Set 1":>15s}{"Energy":^15s}{"Cut Size":^15s}')
    print('-' * 60)
    for sample, E in sampleset.data(fields=['sample', 'energy']):
        S0 = [k for k, v in sample.items() if v == 0]
        S1 = [k for k, v in sample.items() if v == 1]
        print(f'{str(S0):>15s}{str(S1):>15s}{str(E):^15s}{str(int(-1 * E)):^15s}')


def draw_cut_from_sample(graph: nx.Graph, sample: SampleView) -> None:
    """
    Adapted from the example available on the dwave GitHub profile:

    Available here: https://github.com/dwave-examples/maximum-cut/blob/master/maximum_cut.py
    """
    S0 = [node for node in graph.nodes if not sample[node]]
    S1 = [node for node in graph.nodes if sample[node]]
    cut_edges = [(u, v) for u, v in graph.edges if sample[u] != sample[v]]
    uncut_edges = [(u, v) for u, v in graph.edges if sample[u] == sample[v]]

    pos = nx.spectral_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist=S0, node_color='r')
    nx.draw_networkx_nodes(graph, pos, nodelist=S1, node_color='c')
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(graph, pos, edgelist=uncut_edges, style='solid', width=3)
    nx.draw_networkx_labels(graph, pos)


def __sanitize_df(ground_truth: pd.DataFrame, qubo: QUBO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for idx, row in ground_truth.iterrows():
        x = row.drop('energy').values.astype(int)
        ground_truth.at[idx, 'energy'] = np.round(x @ qubo.qubo_matrix @ x.T, 5)

    for idx, row in qubo.solutions.iterrows():
        x = row.drop('energy').values.astype(int)
        qubo.solutions.at[idx, 'energy'] = np.round(x @ qubo.qubo_matrix @ x.T, 5)

    return ground_truth, qubo.solutions


def compare_solutions(ground_truth: pd.DataFrame, qubo: QUBO) -> None:
    ground_truth, sol = __sanitize_df(ground_truth, qubo)

    log.info('\n' + str(ground_truth['energy'].describe()))
    log.info(f'The best proposed solution has energy {min(sol.energy)}')
