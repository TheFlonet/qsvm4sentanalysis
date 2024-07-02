import logging
from collections import defaultdict
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
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
            if pd.isna(val2) and not pd.isna(val1):
                combined_row.append(val1)
            elif pd.isna(val1) and not pd.isna(val2):
                combined_row.append(val2)
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
    return __fill_with_nan(pd.Index(all_indices + ['energy']), merge)


def __fill_with_nan(schema: pd.Index, df_to_fill: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(schema) - set(df_to_fill.columns)
    for col in missing_columns:
        df_to_fill[col] = np.nan
    return df_to_fill[schema]


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


def __search_best_row(row: pd.Series, qubo_matrix: np.ndarray) -> Tuple[pd.Series, int]:
    nan_indices = row.index[row.isna()]
    n = len(nan_indices)
    best_row = row.copy()
    min_energy = np.inf
    trials = 0

    for i in range(2 ** n):
        trials += 1
        binary_combination = [int(x) for x in list(f'{i:0{n}b}')]
        temp_row = row.copy()
        temp_row[nan_indices] = binary_combination
        energy = temp_row.values @ qubo_matrix @ temp_row.values.T
        if energy < min_energy:
            min_energy = energy
            best_row = temp_row

    best_row.loc['energy'] = min_energy
    return best_row, trials


def __brute_force(df: pd.DataFrame, qubo_matrix: np.ndarray) -> Tuple[pd.DataFrame, int]:
    result_df = df.copy()
    trials = 0
    for idx, row in result_df.drop(columns=['energy']).iterrows():
        if row.hasnans:
            new_row, t = __search_best_row(row, qubo_matrix)
            trials += t
            result_df.iloc[idx] = new_row

    return result_df.reset_index(drop=True), trials


def __aggregate_solutions(solutions: List[QUBO], qubo: QUBO) -> QUBO:
    # Aggregate upper-left qubo with lower-right
    starting_sols = __combine_ul_lr(solutions[0], solutions[2])
    # Set missing columns in upper-right qubo to NaN
    ur_qubo_filled = __fill_with_nan(starting_sols.columns, solutions[1].solutions)
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
            data = [[np.nan for _ in range(len(all_indices) + 1)]]
            qubo.solutions = pd.DataFrame(data, columns=all_indices + ['energy'])
        else:
            res = (sampler.sample_qubo(qubo.qubo_dict, num_reads=10)
                   .to_pandas_dataframe()
                   .drop(columns=['num_occurrences'])
                   .drop_duplicates()
                   .sort_values(by='energy', ascending=True))
            qubo.solutions = res[res['energy'] == min(res['energy'])]
        return qubo

    sub_problems = __split_problem(qubo, dim)
    sub_problems = [subqubo_solve(sampler, q, dim // 2, cut_dim) for q in sub_problems]
    sol = __aggregate_solutions(sub_problems, qubo)
    return sol


def __sanitize_df(qubo: QUBO) -> pd.DataFrame:
    for idx, row in qubo.solutions.iterrows():
        x = row.drop('energy').values.astype(int)
        qubo.solutions.at[idx, 'energy'] = np.round(x @ qubo.qubo_matrix @ x.T, 5)

    return qubo.solutions


def compare_solutions(min_sol: float, max_sol: float, qubo: QUBO) -> None:
    log.info(f'Ground truth solutions range from {np.round(min_sol, 5)} and {np.round(max_sol, 5)}')
    log.info(f'The best proposed solution has energy {min(__sanitize_df(qubo).energy)}')
