import logging
import numpy as np
import pandas as pd
from subqubo.QUBO import QUBO

log = logging.getLogger('subqubo')


def sanitize_df(qubo: QUBO) -> pd.DataFrame:
    for idx, row in qubo.solutions.iterrows():
        x = row.drop('energy').values.astype(int)
        qubo.solutions.at[idx, 'energy'] = np.round(x @ qubo.qubo_matrix @ x.T, 5)

    return qubo.solutions
