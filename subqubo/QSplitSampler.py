import logging
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd

from subqubo.QUBO import QUBO

log = logging.getLogger('subqubo')


class QSplitSampler:
    def __init__(self, sampler, cut_dim):
        self.sampler = sampler
        self.cut_dim = cut_dim

    def run(self, qubo: QUBO, dim: int) -> QUBO:
        if dim <= self.cut_dim:
            if len(qubo.qubo_dict) == 0:
                all_indices = sorted(list(set(qubo.rows_idx).union(qubo.cols_idx)))
                data = [[np.nan for _ in range(len(all_indices) + 1)]]
                qubo.solutions = pd.DataFrame(data, columns=all_indices + ['energy'])
            else:
                res = (self.sampler.sample_qubo(qubo.qubo_dict, num_reads=10)
                       .to_pandas_dataframe()
                       .drop(columns=['num_occurrences'])
                       .drop_duplicates()
                       .sort_values(by='energy', ascending=True))
                qubo.solutions = res[res['energy'] == min(res['energy'])]
            return qubo

        ul, ur, lr = self.__split_problem(qubo, dim)
        ul = self.run(ul, dim // 2)
        lr = self.run(lr, dim // 2)
        return self.__aggregate_solutions(ul, lr, ur, qubo)

    def __aggregate_solutions(self, ul: QUBO, lr: QUBO, ur: QUBO, qubo: QUBO) -> QUBO:
        # Aggregate upper-left qubo with lower-right
        starting_sols = self.__combine_ul_lr(ul, lr)

        # Fill nans
        starting_sols = self.__local_search(starting_sols, qubo)

        # Add information from upper right matrix
        for i, row in starting_sols.iterrows():
            first, second = np.split(row.drop('energy').values, 2)
            starting_sols.loc[i, 'energy'] += first.T @ ur.qubo_matrix @ second
        qubo.solutions = starting_sols.nsmallest(n=10, columns=['energy'])

        return qubo

    def __local_search(self, df: pd.DataFrame, qubo: QUBO):
        for i, row in df.iterrows():
            no_energy = row.drop('energy')

            if not np.any(np.isnan(no_energy.values)):
                df.loc[i, 'energy'] = no_energy.values.T @ qubo.qubo_matrix @ no_energy.values
            else:
                nans = no_energy[np.isnan(no_energy)]
                qubo_nans = defaultdict(int)
                for row_idx in nans.index:
                    for col_idx in nans.index:
                        qubo_nans[(row_idx, col_idx)] = qubo.qubo_dict.get((row_idx, col_idx), 0)
                nans_sol = self.sampler.sample_qubo(qubo_nans, num_reads=10)
                nans_sol = nans_sol.to_pandas_dataframe().sort_values(by='energy', ascending=True).iloc[0]
                df.loc[i, nans.index] = nans_sol.drop('energy')
                df.loc[i, 'energy'] = nans_sol['energy']

        return df

    @staticmethod
    def __fill_with_nan(schema: pd.Index, df_to_fill: pd.DataFrame) -> pd.DataFrame:
        missing_columns = set(schema) - set(df_to_fill.columns)
        for col in missing_columns:
            df_to_fill[col] = np.nan
        return df_to_fill[schema]

    def __combine_ul_lr(self, ul: QUBO, lr: QUBO) -> pd.DataFrame:
        all_indices = sorted(list(set(ul.rows_idx).union(lr.cols_idx)))
        ul.solutions['tmp'] = 1
        lr.solutions['tmp'] = 1
        merge = pd.merge(ul.solutions, lr.solutions, on='tmp')
        merge['energy'] = merge['energy_x'] + merge['energy_y']
        merge = merge.drop(['energy_x', 'energy_y', 'tmp'], axis=1)
        ul.solutions.drop('tmp', axis=1, inplace=True)
        lr.solutions.drop('tmp', axis=1, inplace=True)
        return self.__fill_with_nan(pd.Index(all_indices + ['energy']), merge)

    @staticmethod
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
        split_l_r = defaultdict(int)
        split_idx = dim // 2

        for k, v in qubo.qubo_dict.items():
            row, col = k[0] - 1, k[1] - 1
            if row < split_idx and col < split_idx:
                split_u_l[k] = v
            elif row < split_idx <= col:
                split_u_r[k] = v
            elif row >= split_idx and col >= split_idx:
                split_l_r[k] = v
            else:
                raise ValueError(
                    'All values in the lower left matrix should be 0, so not present in the qubo dictionary')

        res = (QUBO(split_u_l, cols_idx=qubo.cols_idx[:split_idx], rows_idx=qubo.rows_idx[:split_idx]),
               QUBO(split_u_r, cols_idx=qubo.cols_idx[split_idx:], rows_idx=qubo.rows_idx[:split_idx]),
               QUBO(split_l_r, cols_idx=qubo.cols_idx[split_idx:], rows_idx=qubo.rows_idx[split_idx:]))

        return res
