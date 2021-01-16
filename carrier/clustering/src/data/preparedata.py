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
# https://github.com/owid/covid-19-data
covid_data: pd.DataFrame = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
covid_data_g: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")
cogid_data_dict = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-codebook.csv")


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
# %% [markdown]
'''
### Collect less frequent categories
'''
# TODO: Collect less frequent categories

# %%
