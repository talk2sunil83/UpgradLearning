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

metadata = pd.read_csv(RAW_DATA_PATH/'metadata.zip')


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
# %%
eu.print_value_count_percents(metadata[['journal']])
# %% [markdown]
'''
#### value counts plots
'''
# %%
eu.plot_univariate_categorical_columns(metadata[['license']], x_rotation=90, plot_limit=50)
# %%
eu.plot_univariate_categorical_columns(metadata[['license']], interactive=True, plot_limit=50, log_y=True)

# %%
eu.plot_univariate_categorical_columns(metadata[['journal']], x_rotation=90, plot_limit=50)
# %%
eu.plot_univariate_categorical_columns(metadata[['journal']], interactive=True, plot_limit=50, log_y=True)

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
sorted(list(metadata[metadata[publish_time].apply(lambda x: str(x).startswith('2021-'))][publish_time]), reverse=True)[:30]
# %%
metadata[publish_time].head()

# %%
metadata[publish_time].tail()
# %%
metadata[publish_time] = pd.to_datetime(metadata[publish_time], format='%Y-%m-%d')

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
metadata[metadata[publish_time].isna() | metadata[publish_year].isna()][[publish_time, publish_year]]

# %%
year_wise_paper_published = metadata.groupby(publish_year)['cord_uid'].count().reset_index()

# %%
plt_frame: pd.DataFrame = year_wise_paper_published.copy()
plt_frame[publish_year] = plt_frame[publish_year].astype(str)
plt_frame[publish_year] = plt_frame[publish_year].apply(lambda x: x.split('.')[0] if len(x) > 0 else x)
plt_frame.columns = [publish_year, 'paper_count']
# %%

plt_frame.plot(x=publish_year,
               y='paper_count', kind='bar', figsize=(20, 10), title="Year wise paper count",
               xlabel="Year", ylabel="Paper count")
plt.show()
# %%

plt_frame.plot(x=publish_year,
               y='paper_count', kind='bar', figsize=(20, 10), title="Year wise paper count", logy=True,
               xlabel="Year", ylabel="Paper count on (Log Scale)")
plt.show()
# eu.plot_two_variables(plt_frame, publish_year, 'paper_count')
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
# %% [markdown]
'''
### Bi-variate
'''
# %% [markdown]
'''
### Correlation
'''

# Mostly positive correlated data
# %% [markdown]
'''
#### Numeric-Numeric (Scatter plot)
'''

# %% [markdown]
'''
####  Numeric-Categorical (Box and violin)
'''


# %% [markdown]
'''
#### Categorical-Categorical (Cross Table)
'''
# %%

# pd.crosstab(metadata[publish_year], metadata[['journal', 'license']], rownames=["Publish Year"], colnames=["Journal, 'License"])
pd.crosstab(metadata['journal'], metadata['license'])
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
