# %%
import set_base_path
import numpy as np
import pandas as pd
from IPython.display import display
import plotly.figure_factory as ff
import plotly.graph_objects as go
from enum import Enum, auto
from typing import List, Sequence, Tuple
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from src.constants import RAW_DATA_PATH, INTERIM_DATA_PATH
import src.utils.preprocess as pp
import src.utils.eda as eu
import src.utils.common as cmn
warnings.filterwarnings('ignore')

# %%
# Ignore warnings

# %% [markdown]
'''
## Data load
'''
# %%
# Load Data
metadata: pd.DataFrame = pd.read_csv(RAW_DATA_PATH/'metadata.zip')


# %% [markdown]
'''
## Pandas settings
'''
# %%
pd.options.display.max_columns = None
pd.options.display.max_rows = 500
# pd.options.display.width = None
# pd.options.display.max_colwidth = 100
pd.options.display.precision = 3
# %%
# %% [markdown]
'''
### Collect less frequent categories
'''
# TODO: Collect less frequent categories

# %%
# metadata.head().style.set_properties(**{'text-align': 'left'}).hide_index()
metadata.head(1).style.hide_index()
# %%
metadata.shape
# %%
metadata.isnull().sum()

# %%
eu.print_null_percents(metadata)
# %%
eu.count_of_uniques(metadata, display_res=True)

# %%
metadata.drop("mag_id", axis=1, inplace=True)

# %%
cols_with_nulls = eu.print_null_percents(metadata)
# %%
replace_value_map = {c: "NOT_PROVIDED" for c in cols_with_nulls.index}
# %%
for col, value in replace_value_map.items():
    metadata[col] = metadata[col].fillna(value)
# %%

eu.print_null_percents(metadata)
# %%
metadata.to_csv(INTERIM_DATA_PATH/'metadata.zip', index=False, quoting=1, compression='zip')

# %%
