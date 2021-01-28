from src.constants import NUM_CORES
import pandas as pd
import numpy as np
from multiprocessing.pool import Pool


def parallelize_dataframe(df: pd.DataFrame, func, n_cores=NUM_CORES):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
