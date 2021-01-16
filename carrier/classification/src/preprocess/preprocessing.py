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


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import date
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
from src.constants import *
from sklearn.impute import IterativeImputer
from src.utils import preprocess as pp
from src.utils import eda as eu
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
# merged_df = pd.read_csv(INTERIM_DATA_PATH / "merged_df.csv")
claims_with_amount: pd.DataFrame = pd.read_feather(RAW_DATA_PATH / "claims_with_amount.feather")
labour: pd.DataFrame = pd.read_feather(RAW_DATA_PATH / "labour.feather")
parts_replaced: pd.DataFrame = pd.read_feather(RAW_DATA_PATH / "parts_replaced.feather")
# %% [markdown]
'''
## Pandas settings
'''
# %%
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3


# %% [markdown]
'''
##  Pre processing
'''


# %% [markdown]
'''
### Collect less frequent categories
'''
# %%
labour = pp.replace_values_having_less_count(labour, [JOB_CODE])
parts_replaced = pp.replace_values_having_less_count(parts_replaced, [PART_TYPE, INS_PART_CODE])
claims_with_amount = pp.replace_values_having_less_count(claims_with_amount, CLAIM_CAT_COLS)
# %%


# %%
labour[STD_LABOR_HRS] = labour[STD_LABOR_HRS].fillna(0.)
labour[ADD_LABOR_HRS] = labour[ADD_LABOR_HRS].fillna(0.)
labour["LABOUR"] = labour[STD_LABOR_HRS] + labour[ADD_LABOR_HRS]

# %%
new_labours = labour.pivot(columns=JOB_CODE, values='LABOUR')
new_labours[CLAIM_NUMBER] = labour[CLAIM_NUMBER]
# new_labours["CLAIM_ID"] = labour['CLAIM_ID']
new_labours = new_labours.drop_duplicates(subset=[CLAIM_NUMBER], keep='first')
new_labours = new_labours.fillna(0.)
# %%
new_parts_replaced = parts_replaced.pivot(columns=INS_PART_CODE, values=INS_PART_QNTY)
print(new_parts_replaced.shape)
new_parts_replaced[CLAIM_NUMBER] = parts_replaced[CLAIM_NUMBER]
# new_parts_replaced["CLAIM_ID"] = parts_replaced['CLAIM_ID']
new_parts_replaced = new_parts_replaced.drop_duplicates(subset=[CLAIM_NUMBER], keep='first')

# %%
new_parts_replaced = new_parts_replaced.fillna(0.)
# %%
claims_with_amount[REG_PRODUCT_FAMILY_NAME] = claims_with_amount[REG_PRODUCT_FAMILY_NAME].astype(str)
# %%
# Merge all dataframes
merged_df = pd.merge(pd.merge(claims_with_amount, new_labours, how='inner', on=CLAIM_NUMBER), new_parts_replaced, how='inner', on=CLAIM_NUMBER)

# %%
# Check the columns having less than 100 values
# %%
pivoted_columns = list(labour[JOB_CODE].unique()) + list(parts_replaced[INS_PART_CODE].unique())
# %%
zeros = merged_df[pivoted_columns] == 0
zero_percentage = (((zeros).sum()*100)/merged_df.shape[0]).sort_values(ascending=False)
zero_valued_columns = zero_percentage[zero_percentage == 100.].index
zero_valued_columns

# %%
print(f"Shape before drop: {merged_df.shape}")
merged_df = merged_df.drop(zero_valued_columns, axis=1)
print(f"Shape after drop: {merged_df.shape}")

# %%
#  Prepare y
target_col_name = 'IS_SUSPECTED'
predicted = pd.read_feather(RAW_DATA_PATH/'Batch_Predicted.feather')
predicted = predicted[['CLAIM_NUMBER', target_col_name]]
predicted[target_col_name] = predicted[target_col_name] > 0.5
# %%
print(merged_df.shape)
merged_df = merged_df.merge(predicted, on='CLAIM_NUMBER', how='inner')
print(target_col_name in merged_df.columns)
print(merged_df.shape)

# %% [markdown]
'''
Column Dropping
'''

# %%
for col in ['DEALER_NAME', 'PARENT_DEALER', 'CLAIM_ID', 'CLAIM_NUMBER', 'PRODUCT_NAME.1', 'SERIAL_NUMBER', 'DEALER_NUMBER']:
    merged_df.drop(col, axis=1, inplace=True)
# %%
date_cols = [c for c in merged_df.columns if c.endswith("_DATE")]
# %%
merged_df = pp.get_days_from_date(merged_df, date_cols)
# %%
# Single valued columns
single_valued_cols = pp.get_single_valued_columns(merged_df)
print(single_valued_cols)
merged_df.drop(single_valued_cols, axis=1, inplace=True)
# %% [markdown]
'''
###  Null Handling
'''

# %%
DAYS_SINCE_WARNTY_END = 'DAYS_SINCE_WARNTY_END'
DAYS_SINCE_WARNTY_START = 'DAYS_SINCE_WARNTY_START'
UNITS_USAGE = 'UNITS_USAGE'
APPLICABLE_POLICY = 'APPLICABLE_POLICY'
CAUSAL_REG_PART = 'CAUSAL_REG_PART'

# %%
null_fill_map = dict(
    DAYS_SINCE_WARNTY_END=merged_df[DAYS_SINCE_WARNTY_END].mean(), DAYS_SINCE_WARNTY_START=merged_df[DAYS_SINCE_WARNTY_START].mean(), UNITS_USAGE=merged_df[UNITS_USAGE].mean(), APPLICABLE_POLICY="OTHER_APPLICABLE_POLICY", CAUSAL_REG_PART="OTHER_CAUSAL_REG_PART"
)
# %%
for col, value in null_fill_map.items():
    merged_df[col] = merged_df[col].fillna(value)

eu.print_null_percents(merged_df)
# %% [markdown]
'''
###  Conversion of categorical (OHE or mean)
'''
# %%
# null_counts = merged_df.isnull().sum()

object_cols = merged_df.select_dtypes('object').columns
object_cols


# %%
merged_df = pp.get_dummies_for_col(merged_df, object_cols)

# %%
new_col_names = []
for i, col_name in enumerate(merged_df.columns):
    if col_name != target_col_name:
        new_col_names.append(f"{col_name}_{str(i)}")
# %%
merged_df.columns = new_col_names + [target_col_name]
# %%
merged_df.to_feather(INTERIM_DATA_PATH/"merged_df.feather")
# %%
# pd.get_dummies(merged_df, drop_first=True, columns=object_cols)
# %% [markdown]
'''
###  Scaling
'''
# %%
# merged_df = pd.read_feather(INTERIM_DATA_PATH/"merged_df.feather")
target = merged_df.pop(target_col_name)

# %%
ss = StandardScaler()
merged_df_scaled = pd.DataFrame(ss.fit_transform(merged_df), columns=merged_df.columns)

merged_df_scaled[target_col_name] = target
merged_df_scaled.to_feather(INTERIM_DATA_PATH/"merged_df_scaled.feather")


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
# %%
pca = PCA(0.9)
merged_df_dimred = pd.DataFrame(pca.fit_transform(merged_df_scaled))
merged_df_dimred.columns = [str(col) for col in merged_df_dimred.columns]
merged_df_dimred[target_col_name] = target
merged_df_dimred.to_feather(INTERIM_DATA_PATH/"merged_df_dimred.feather")
# %%
pca3 = PCA(3)
merged_df_dimred_pca3 = pd.DataFrame(pca3.fit_transform(merged_df_scaled))
merged_df_dimred_pca3.columns = [str(col) for col in merged_df_dimred_pca3.columns]
merged_df_dimred_pca3.to_feather(INTERIM_DATA_PATH/"merged_df_dimred_pca3.feather")
# %%
pca2 = PCA(2)
merged_df_dimred_pca2 = pd.DataFrame(pca2.fit_transform(merged_df_scaled))
merged_df_dimred_pca2.columns = [str(col) for col in merged_df_dimred_pca2.columns]
merged_df_dimred_pca2.to_feather(INTERIM_DATA_PATH/"merged_df_dimred_pca2.feather")

# %%
fig = px.scatter_3d(merged_df_dimred_pca3, x='0', y='1', z='2', width=1000, height=1000, size_max=2, opacity=0.7)
fig.show()
# %%
fig = px.scatter(merged_df_dimred_pca3, x='0', y='1', width=1000, height=1000, size_max=2, opacity=0.7)
fig.show()

# %%
# merged_df_scaled = pd.read_feather(INTERIM_DATA_PATH/"merged_df_scaled.feather")
# merged_df = pd.read_feather(INTERIM_DATA_PATH/"merged_df.feather")
# merged_df_dimred = pd.read_feather(INTERIM_DATA_PATH/"merged_df_dimred.feather")
# %%
# %% [markdown]
'''
Get Target variables
Modeling
Tuning
feature importance

'''
