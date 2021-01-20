# %% [markdown]
'''
# Calculate suspect score for manufacturing claims
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

import set_base_path
import src.utils.eda as eu
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

metadata = pd.read_csv(RAW_DATA_PATH/'metadata.csv')


# %% [markdown]
'''
## Pandas settings
'''
# %%
pd.options.display.max_columns = 300
pd.options.display.max_rows = 300
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3

# %% [markdown]
'''
# EDA
'''

# %% [markdown]
'''
## Data Overview
'''

# %%
eu.get_data_frame_overview(metadata, 2)

# %%
# %% [markdown]
'''
### Univariate
'''

# %% [markdown]
'''
#### value counts
'''
# %%
eu.print_value_count_percents(metadata[['license']])
# %% [markdown]
'''
#### value counts plots
'''
# %%
eu.plot_univariate_categorical_columns(metadata[['license']], x_rotation=90, plot_limit=50)
# %%
eu.plot_univariate_categorical_columns(metadata[['license']], interactive=True, plot_limit=50, log_y=True)
# %% [markdown]
'''
#### distributions
'''
# %%
metadata.dtypes
# %%
metadata = metadata.convert_dtypes()
# %%
metadata.dtypes

# %%
publish_time = 'publish_time'

# %%
metadata[publish_time].head()

# %%
metadata[publish_time].tail()
# %%

metadata[publish_time] = pd.to_datetime(metadata[publish_time], infer_datetime_format=True)

# %%
metadata[publish_time].head()

# %%
metadata[publish_time].tail()
# %%
publish_year = 'publish_year'
publish_month = 'publish_month'
publish_day = 'publish_day'

# %%
pub_dates: pd.Series = metadata[publish_time].dt
metadata[publish_year] = pub_dates.year
metadata[publish_month] = pub_dates.month
metadata[publish_day] = pub_dates.dayofweek


# %%

# metadata.groupby(pd.Grouper(key="publish_time", freq="1M"))['cord_uid'].count()
metadata.groupby(publish_year)['cord_uid'].count()

# %% [markdown]
'''
## Drop unwanted columns
'''

# %% [markdown]
'''
## Fix column dtypes
'''
# %% [markdown]
'''
#### Plotting numeric and categorical
'''
# %%
num_cols, CLAIM_CAT_COLS

# %%
len(num_cols), len(CLAIM_CAT_COLS)
# %% [markdown]
'''
### Bi-variate
'''
# %% [markdown]
'''
### Correlation
'''
# %%
plt.figure(figsize=(10, 10))
sns.heatmap(claims_with_amount[num_cols].corr(), annot=True)
plt.show()

# Mostly positive correlated data
# %% [markdown]
'''
#### Numeric-Numeric (Scatter plot)
'''

# %%
eu.plot_two_variables(claims_with_amount, 'CLAIMED_AMOUNT', 'CLAIM_PAID_AMOUNT')
# %%
plt.figure(figsize=(10, 10))
eu.plot_two_variables(claims_with_amount, 'UNITS_USAGE', 'CLAIM_PAID_AMOUNT')
# %% [markdown]
'''
####  Numeric-Categorical (Box and violin)
'''

# %%
new_cols_cat = CLAIM_CAT_COLS[:]
for rem_col in ["DEALER_NUMBER", "CAUSAL_REG_PART", "DEALER_CITY", "DEALER_STATE", "FAULT_LOCN", "FAULT_CODE"]:
    new_cols_cat.remove(rem_col)
for col in new_cols_cat:
    plt.figure(figsize=(35, 10))
    print(f"\nPlotting {col} vs CLAIM_PAID_AMOUNT\n")
    eu.plot_two_variables(claims_with_amount, col, 'CLAIM_PAID_AMOUNT', x_rotation=90, legend=False)
# %% [markdown]
'''
#### Categorical-Categorical (Cross Table)
'''
# %%
pd.crosstab(claims_with_amount['CLAIM_TYPE'], claims_with_amount['CLAIM_STATE'])
# %%
# TODO: Not working need to check data types
pd.crosstab(claims_with_amount['CLAIM_TYPE'], claims_with_amount[['CLAIM_STATE', 'APPLICABLE_POLICY',
                                                                  'DEALER_NUMBER',
                                                                  'DEALER_CITY',
                                                                  'DEALER_STATE',
                                                                  'DEALER_COUNTRY',
                                                                  'CAUSAL_REG_PART',
                                                                  'FAULT_CODE',
                                                                  'FAULT_LOCN',
                                                                  'REG_PRODUCT_FAMILY_NAME',
                                                                  'REG_SERIES_NAME',
                                                                  'MODEL_NAME',
                                                                  'REG_MODEL_CODE',
                                                                  'VARIANT']])
# %% [markdown]
'''
Print a data frame with color
'''

# %%
'''
Drop columns
    Single valued
    
Drop Rows

'''
