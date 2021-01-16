# %% [markdown]
'''
# <Title>
'''

# %% [markdown]
'''
# Problem statement
'''

# %% [markdown]
'''
**Author** : Sunil Yadav || yadav.sunil83@gmail.com || +91 96206 38383 || 
'''

# %% [markdown]
'''
# Solution Approach

 - Step1
 - Step2
'''
# %% [markdown]
'''
# Solution
'''

# %% [markdown]
'''
## Lib Imports
'''

# %%

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
warnings.filterwarnings('ignore')


# %%
# Ignore warnings

# %% [markdown]
'''
## Data load
'''
# %%
# Load Data
data_frame = pd.read_csv("a.csv")

# %% [markdown]
'''
## Pandas settings
'''
# %%
pd.options.display.max_columns = None
pd.options.display.max_rows = 500
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3


# %% [markdown]
'''
##  Pre processing
'''
# %% [markdown]
'''
## Pre processing Utils
'''
# %%
# def replace_values_having_less_count(dataframe: pd.DataFrame, target_cols: Sequence[str], threshold: Union[int, Sequence[int]] = 100,  replace_with: Union[int, float, str, Sequence[int], Sequence[float], Sequence[str]] = "OT") -> DataFrame:
#     pass


def replace_values_having_less_count(dataframe: pd.DataFrame, target_cols: Sequence[str], threshold: int = 100,  replace_with="OT") -> pd.DataFrame:
    for c in target_cols:
        vc = dataframe[c].value_counts()
        replace_dict = {v: replace_with for v in list(vc[vc <= threshold].index)}
        dataframe[c] = dataframe[c].replace(replace_dict)
        return dataframe


# %% [markdown]
'''
###  Null Handling
'''
# %% [markdown]
'''
###  Scaling
'''
# %% [markdown]
'''
###  Conversion of categorical (OHE or mean)
'''
# %% [markdown]
'''
###  Outlier Treatment
'''
# %% [markdown]
'''
###  Single valued removal
'''
# %% [markdown]
'''
###  ID Removal
'''
# %% [markdown]
'''
###  Non important column removal
'''
# %% [markdown]
'''
###  Feature creation
'''
# %% [markdown]
'''
####  Data Based
'''
# %% [markdown]
'''
####  Domain based
'''
# %% [markdown]
'''
###  Dimensionality Reduction
'''
