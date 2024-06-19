import itertools
import logging
from collections import defaultdict
from typing import List, Mapping, Hashable, Tuple, Any
import networkx as nx
import numpy as np
import pandas as pd
from dimod import SampleSet
from dimod.views.samples import SampleView
from dwave.samplers import SimulatedAnnealingSampler
from numpy import integer, floating
from scipy.linalg import lu

log = logging.getLogger('subqubo')


def __split_problem(qubo: Mapping[tuple[Hashable, Hashable], float | floating | integer], dim: int) -> Tuple[
    Mapping[tuple[Hashable, Hashable], float | floating | integer],
    Mapping[tuple[Hashable, Hashable], float | floating | integer],
    Mapping[tuple[Hashable, Hashable], float | floating | integer]
]:
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
    qubo_matrix_u_r = np.zeros((split_idx, split_idx))

    for k, v in qubo.items():
        row, col = k[0] - 1, k[1] - 1
        if row < split_idx and col < split_idx:
            split_u_l[k] = v
        elif row < split_idx <= col:
            qubo_matrix_u_r[row % split_idx, col % split_idx] = v
        elif row >= split_idx and col >= split_idx:
            split_d_r[k] = v
        else:
            raise ValueError('All values in the lower left matrix should be 0, so not present in the qubo dictionary')

    qubo_matrix_u_r = lu(qubo_matrix_u_r, permute_l=True)[1]
    for i in range(qubo_matrix_u_r.shape[0]):
        for j in range(qubo_matrix_u_r.shape[1]):
            if qubo_matrix_u_r[i, j] != 0:
                split_u_r[(i + 1, j + 1 + split_idx)] = qubo_matrix_u_r[i, j]

    return split_u_l, split_u_r, split_d_r


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


def __combine_ul_lr(ul_qubo: pd.DataFrame, lr_qubo: pd.DataFrame) -> pd.DataFrame:
    ul_qubo['tmp'] = 1
    lr_qubo['tmp'] = 1
    merge = pd.merge(ul_qubo, lr_qubo, on='tmp')
    merge['energy'] = merge['energy_x'] + merge['energy_y']
    return merge.drop(['energy_x', 'energy_y', 'tmp'], axis=1)


def __fill_ur_qubo(schema: pd.DataFrame, df_to_fill: pd.DataFrame) -> pd.DataFrame:
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


def __brute_force(df: pd.DataFrame, qubo_matrix: np.ndarray) -> Tuple[pd.DataFrame, int]:
    result_rows = []
    for idx, row in df.drop(columns=['energy']).iterrows():
        if row.isna().sum() == 0:
            new_row = row.copy()
            new_row['energy'] = df.loc[idx, 'energy']
            result_rows.append(new_row)
        else:
            nan_columns = row.index[row.isna()]
            for combination in itertools.product([0, 1], repeat=len(nan_columns)):
                new_row = row.copy()
                new_row[nan_columns] = combination
                new_row['energy'] = np.nan
                result_rows.append(new_row)

    result_df = pd.DataFrame(result_rows, columns=df.columns).reset_index(drop=True)
    result_df = result_df.drop_duplicates(subset=result_df.columns[:-1]).reset_index(drop=True)

    trials = 0
    for idx, row in result_df.iterrows():
        if pd.isna(row['energy']):
            trials += 1
            x = row.drop('energy').values.astype(int)
            result_df.at[idx, 'energy'] = x @ qubo_matrix @ x.T

    return result_df.reset_index(drop=True), trials


def __aggregate_solutions(solutions: List[pd.DataFrame], qubo_matrix: np.ndarray) -> pd.DataFrame:
    # Aggregate upper-left qubo with lower-right
    starting_sols = __combine_ul_lr(solutions[0], solutions[2])
    # Set missing columns in upper-right qubo to NaN
    ur_qubo_filled = __fill_ur_qubo(starting_sols, solutions[1])
    # Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
    closest_df = __get_closest_assignments(starting_sols, ur_qubo_filled)

    # Combine
    combined_df = pd.DataFrame([__combine_rows(row1, row2) for (_, row1), (_, row2) in
                                zip(starting_sols.iterrows(), closest_df.iterrows())],
                               columns=starting_sols.columns)

    # Brute force resolution
    res, trials = __brute_force(combined_df, qubo_matrix)
    log.info(f'Dimension {qubo_matrix.shape[0]}. Conflicts resolved with {trials} classic resolutions' if trials > 0
             else f'Dimension {qubo_matrix.shape[0]}. No conflict, merge successfully done')

    return res


def subqubo_solve(sampler: SimulatedAnnealingSampler,
                  qubo: Mapping[tuple[Hashable, Hashable], float | floating | integer],
                  dim: int) -> pd.DataFrame:
    if dim == 2:
        if len(qubo) == 0:
            return pd.DataFrame({'energy': [0]})
        res = (sampler.sample_qubo(qubo, num_reads=10)
               .to_pandas_dataframe()
               .drop(columns=['num_occurrences'])
               .drop_duplicates()
               .sort_values(by='energy', ascending=True))
        return res[res['energy'] == min(res['energy'])]

    qubo_matrix = np.zeros((dim, dim))
    for k, v in qubo.items():
        qubo_matrix[(k[0] - 1) % dim, (k[1] - 1) % dim] = v

    return __aggregate_solutions([subqubo_solve(sampler, q, dim // 2) for q in __split_problem(qubo, dim)], qubo_matrix)


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


def __check_dataframe_consistency(ground_truth: pd.DataFrame, sol: pd.DataFrame,
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
    ground_truth, sol = __check_dataframe_consistency(ground_truth, sol, qubo_matrix)

    log.info(f'The best ground truth solution has energy {min(ground_truth.energy)}')
    log.info(f'The best proposed solution has energy {min(sol.energy)}')
    if min(ground_truth.energy) == 0:
        gap = ((min(sol.energy) + 1 - min(ground_truth.energy) + 1) / abs(min(ground_truth.energy) + 1)) * 100
    else:
        gap = ((min(sol.energy) - min(ground_truth.energy)) / abs(min(ground_truth.energy))) * 100
    log.info(f'Relativa gap: {gap:.2f}%')
