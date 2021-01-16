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
## Data Overview
'''

# %%
# Shape
data_frame.shape
# %%
# DTypes
data_frame.dtypes
# %%
# Total Nulls
data_frame.isnull().sum().sum()
# %%
# Nulls
round(data_frame.isnull().mean()*100, 2)
# %%
# Duplicate

len(data_frame) - len(data_frame.drop_duplicates())
# %%
# Sample
data_frame.sample(1)
# %%
# Head
data_frame.head(1)
# %%
# tail
data_frame.tail(1)
# %%
# Describe
data_frame.describe(include='all').T

# %%
# info
data_frame.info(verbose=1)
# %%
# Count of data types
data_frame.dtypes.value_counts()
# %% [markdown]
'''
## EDA
'''
# %% [markdown]
'''
### EDA Utility Functions
'''

# %%


def print_null_percents(frame: pd.DataFrame, full: bool = False, display_cols=True):
    """Prints null columns perdent and count

    Args:
        frame (pd.DataFrame):Dataframe where null needs to be counted
        full (bool, optional): show all columns. Defaults to False.
        display_cols (bool, optional): show columns or not. Defaults to True.
    """
    null_counts = frame.isna().sum()
    if not full:
        null_counts = null_counts[null_counts > 0]
    if display_cols:
        display(round((null_counts/frame.shape[0])*100, 2).sort_values(ascending=False))
    print(f"Columns count with null: {len(null_counts[null_counts > 0])}")


class GraphType(Enum):
    """Graph Type Enum

    Args:
        Enum ([type]): Built-in Enum Class
    """
    BAR = auto()
    LINE = auto()
    DIST = auto()


def plot_univariate_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        graph_type: GraphType = None,
        showlegend: bool = False,
        log_x=False,
        log_y=False,
        * args,
        **kwargs) -> None:
    """Bar plots a interger series

    Args:
        series (pd.Series): series to be plotted
        title (str): graph title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        graph_type (GraphType, optional): graph type
        showlegend (bool, optional): default False
        log_x (bool, optional): default False
        log_y (bool, optional): default False
    """
    labels = {"x": xlabel, "y": ylabel}
    fig = None
    if graph_type is None or graph_type == GraphType.BAR:
        fig = px.bar(x=series.index, y=series, color=series.index,
                     title=title, labels=labels, log_x=log_x, log_y=log_y, *args, **kwargs)

    if graph_type == GraphType.LINE:
        px.scatter(x=series.index, y=series, title=title, labels=labels, color=series.index, *args,
                   **kwargs)

    fig.update_layout(showlegend=showlegend)
    fig.show()


def get_univariate_cat_plot_strs(value: str, **kwargs) -> Tuple[str, str, str]:
    """Creates graph title, x-axis text and y-axis text for given value

    Args:
        value (str): column name

    Returns:
        Tuple[str, str, str]: title, x-axis text and y-axis text
    """
    full_name = value  # TODO: write logic to make name
    if len(full_name) > 30:
        full_name = value
    count_str = full_name + ' Count' + " - Log Scale" if kwargs.get("log_y") else ""
    return count_str + ' Plot', full_name, count_str


def plot_cat_data(c: str, value_counts_ser: pd.Series, *args, **kwargs):
    """Plots the value count series

    Args:
        c ([str]): column name
        value_counts_ser ([pd.Series]): value counts series
    """
    t, xl, yl = get_univariate_cat_plot_strs(c, **kwargs)
    plot_univariate_series(value_counts_ser, t, xl, yl, *args, **kwargs)


def plot_univariate_categorical_columns(categorical_cols: Sequence[str], dataframe: pd.DataFrame, plot_limit: int = 30, print_value_counts=False, *args, **kwargs) -> None:
    """plots categorical variable bars

    Args:
        categorical_cols (Sequence[str]): categorical columns
        dataframe (pd.DataFrame): DataFrame
    """
    for c in categorical_cols:
        value_counts_ser = dataframe[c].value_counts()
        if print_value_counts:
            print(value_counts_ser)
        cnt_len = len(value_counts_ser)
        if cnt_len > 1 and cnt_len < plot_limit:
            plot_cat_data(c, value_counts_ser, *args, **kwargs)


def plot_dist(data_frame: pd.DataFrame, cols_to_plot: List[str], merge_all: bool = False, width=800, *args, **kwargs) -> None:

    if merge_all:
        fig = ff.create_distplot(hist_data=data_frame, group_labels=cols_to_plot, *args, **kwargs)
        fig.update_layout(title_text=f"Dist plot for Numeric Columns", width=width)
        fig.show()
    else:
        for _, c in enumerate(cols_to_plot):
            fig = ff.create_distplot(hist_data=[data_frame[c].values], group_labels=[c], *args, **kwargs)
            fig.update_layout(title_text=f"Distribution plot for {c}", width=width)
            fig.show()


def plot_box(df: pd.DataFrame, x: str, y: str) -> None:
    fig = px.box(df, x=x, y=y, color=x)
    fig.show()


def getdtype(col_data: pd.Series):
    if col_data.dtype == np.int64 or col_data.dtype == np.float64:
        return 'num'
    elif col_data.dtype == 'category':
        return 'cat'


def plot_two_variables(df, x, y):
    if getdtype(df[x]) == 'num' and getdtype(df[y]) == 'num':
        fig = px.scatter(df, x=x, y=y, trendline="ols")
        fig.show()
    elif (getdtype(df[x]) == 'cat' and getdtype(df[y]) == 'num'):
        plot_box(df, x, y)
    elif (getdtype(df[x]) == 'num' and getdtype(df[y]) == 'cat'):
        plot_box(df, y, x)


def set_value_count_color(value):
    return "background-color: rgba(221, 207, 155, 0.1)" if value <= 5. else ''


def print_value_count_percents(categorical_cols: Sequence[str], dataframe: pd.DataFrame) -> None:
    total_recs = dataframe.shape[0]
    # ret_values = {}
    for c in categorical_cols:
        value_counts_ser = dataframe[c].value_counts()
        value_counts_per = round(dataframe[c].value_counts()*100/total_recs, 2)
        df = pd.DataFrame({"Value": value_counts_ser.index, "Value Counts": value_counts_ser.values, "Percent": value_counts_per.values})
        df.sort_values(by="Percent", ascending=False)
        # ret_values[c] = df
        print(f"\nValue Counts for {c}")
        # styled_df = df.style.apply(lambda row: highlight_other_group(row, col_count, 5), axis=1)
        styled_df = df.style.format({
            "Percent": "{:.2f}%"
        }). \
            applymap(set_value_count_color, subset=["Percent"]). \
            hide_index()

        display(styled_df)

    # return ret_values


def print_count_of_uniques(dataframe: pd.DataFrame, display_res=False) -> pd.DataFrame:
    cols = dataframe.columns
    unique_values = []
    unique_len = []
    for c in cols:
        uniques = dataframe[c].unique()
        unique_values.append(sorted(uniques))
        unique_len.append(len(uniques))

    frame = pd.DataFrame({
        "Column": cols,
        "Unique Values": unique_values,
        "Column Unique Count": unique_len})
    frame.sort_values(by=["Column Unique Count", "Column"], ascending=[False, True], inplace=True)
    if display_res:
        display(frame.style.hide_index())
    return frame


# %% [markdown]
'''
### Univariate
'''
# %% [markdown]
'''
#### value counts
'''
# %% [markdown]
'''
#### distributions
'''
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
'''
Print a data frame with color
'''


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

# %% [markdown]
'''
## Modeling
'''
# %% [markdown]
'''
### Train test split
'''

# %% [markdown]
'''
### Class Balancing
'''

# %% [markdown]
'''
### Create model
'''
# %% [markdown]
'''
### With Dimensionality Reduction
'''

# %% [markdown]
'''
#### Model 1
'''
# %% [markdown]
'''
### Evaluate
'''
# %% [markdown]
'''
#### On train
'''
# %% [markdown]
'''
#### On test data
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Record performance
'''
# %% [markdown]
'''
#### Model 2
'''
# %% [markdown]
'''
### Evaluate
'''
# %% [markdown]
'''
#### On train
'''
# %% [markdown]
'''
#### On test data
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Record performance
'''
# %% [markdown]
'''
#### Model 3
'''
# %% [markdown]
'''
### Evaluate
'''
# %% [markdown]
'''
#### On train
'''
# %% [markdown]
'''
#### On test data
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Record performance
'''
# %% [markdown]
'''
### Without Dimensionality Reduction multiple models
'''
# %% [markdown]
'''
#### Model 1
'''
# %% [markdown]
'''
### Evaluate
'''
# %% [markdown]
'''
#### On train
'''
# %% [markdown]
'''
#### On test data
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Record performance
'''
# %% [markdown]
'''
#### Model 2
'''
# %% [markdown]
'''
### Evaluate
'''
# %% [markdown]
'''
#### On train
'''
# %% [markdown]
'''
#### On test data
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Record performance
'''
# %% [markdown]
'''
#### Model 3
'''
# %% [markdown]
'''
### Evaluate
'''
# %% [markdown]
'''
#### On train
'''
# %% [markdown]
'''
#### On test data
'''
# %% [markdown]
'''
#### Plot evaluation matrix
'''
# %% [markdown]
'''
#### Record performance
'''
# %% [markdown]
'''
## Best Model Selection
'''
# %% [markdown]
'''
## Conclusion
'''
