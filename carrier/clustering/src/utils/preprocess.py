
from typing import Sequence
import pandas as pd
import numpy as np
from datetime import date
from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool
from typing import Dict
from src.constants import NUM_CORES
from src.utils.common import parallelize_dataframe


def replace_values_having_less_count(dataframe: pd.DataFrame, target_cols: Sequence[str], threshold: int = 100,  replace_with="OTHER") -> pd.DataFrame:
    for c in target_cols:
        vc = dataframe[c].value_counts()
        replace_dict = {v: f"{replace_with}_{c.strip().upper()}S" for v in list(vc[vc <= threshold].index)}
        dataframe[c] = dataframe[c].replace(replace_dict)
    return dataframe


def get_days_from_date(df: pd.DataFrame, date_col_names: Sequence[str]) -> pd.DataFrame:
    current_date = np.datetime64(date.today())
    for c in date_col_names:
        new_col_name = f"DAYS_SINCE_{c.replace('_DATE', '')}"
        df[new_col_name] = (pd.to_datetime(df[c]).astype(np.datetime64) - current_date).dt.days
        df.drop(c, axis=1, inplace=True)
    return df


def get_single_valued_columns(df: pd.DataFrame) -> Sequence[str]:
    return [item[0] for item in list(zip(df.columns, list(map(lambda x: len(df[x].value_counts()), df.columns)))) if item[1] == 1]


def get_dummies_for_col(data_frame: pd.DataFrame, col_names: Sequence[str]):
    for column in col_names:
        data_frame = pd.concat([data_frame, pd.get_dummies(data_frame[column], drop_first=True)], axis=1)
        data_frame.drop(column, axis=1, inplace=True)

    return data_frame


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def __fill_na__(col_values: pd.Series, fill_value) -> pd.Series:
    col_values = col_values.fillna(fill_value)
    return col_values


def fill_na(dataframe: pd.DataFrame, na_map: Dict[str, str]) -> pd.DataFrame:
    # with Parallel(n_jobs=NUM_CORES, require='sharedmem') as parallel:
    parallelize_dataframe(dataframe, __fill_na__, )
    # pass
