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
warnings.filterwarnings('ignore')

# %%
# Ignore warnings

# %% [markdown]
'''
## Data load
'''
# %%
# Load Data
claims_with_amount: pd.DataFrame = pd.read_feather(RAW_DATA_PATH / "claims_with_amount.feather")
labour: pd.DataFrame = pd.read_feather(RAW_DATA_PATH / "labour.feather")
parts_replaced: pd.DataFrame = pd.read_feather(RAW_DATA_PATH / "parts_replaced.feather")

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
# %%
labour.columns

# %% [markdown]
'''
### Collect less frequent categories
'''
# TODO: Collect less frequent categories

# %%
labour['STD_LABOR_HRS'] = labour['STD_LABOR_HRS'].fillna(0.)
labour['ADD_LABOR_HRS'] = labour['ADD_LABOR_HRS'].fillna(0.)
labour["LABOUR"] = labour['STD_LABOR_HRS'] + labour['ADD_LABOR_HRS']

# %%
new_labours = labour.pivot(columns='JOB_CODE', values='LABOUR')
new_labours["CLAIM_NUMBER"] = labour['CLAIM_NUMBER']
# new_labours["CLAIM_ID"] = labour['CLAIM_ID']
new_labours = new_labours.drop_duplicates(subset=['CLAIM_NUMBER'], keep='first')
# new_labours = new_labours.set_index("CLAIM_NUMBER", drop=True)
# %%
# new_labours['CLAIM_ID'].isnull().sum()
# %%
new_labours['CLAIM_NUMBER'].isnull().sum()

# %%
new_labours = new_labours.fillna(0.)

# %%
len(parts_replaced['INS_PART_CODE'].unique())
# %%
len(parts_replaced['CLAIM_NUMBER'].unique())

# %%
new_parts_replaced = parts_replaced.pivot(columns='INS_PART_CODE', values='INS_PART_QNTY')
print(new_parts_replaced.shape)
new_parts_replaced["CLAIM_NUMBER"] = parts_replaced['CLAIM_NUMBER']
# new_parts_replaced["CLAIM_ID"] = parts_replaced['CLAIM_ID']
new_parts_replaced = new_parts_replaced.drop_duplicates(subset=['CLAIM_NUMBER'], keep='first')
print(new_parts_replaced.shape)
# %%
new_parts_replaced.head()
# %%
# new_parts_replaced['CLAIM_ID'].isnull().sum()
# %%
new_parts_replaced['CLAIM_NUMBER'].isnull().sum()

# %%
new_parts_replaced = new_parts_replaced.fillna(0.)
# %%

merged_df = pd.merge(pd.merge(claims_with_amount, new_labours, how='inner', on='CLAIM_NUMBER'), new_parts_replaced, how='inner', on='CLAIM_NUMBER')
merged_df.to_feather(INTERIM_DATA_PATH/"merged_df.feather", index=False, quoting=1)
# %%
merged_df.shape
# %%
