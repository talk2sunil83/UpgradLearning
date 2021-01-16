# %% [markdown]
'''
# [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing) from [Kaggle](https://www.kaggle.com/)
'''
# %% [markdown]
'''
# Problem statement

Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include, for example, the passenger safety cell with crumple zone, the airbag and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium car makers. Daimler’s Mercedes-Benz cars are leaders in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams. .

To ensure the safety and reliability of each and every unique car configuration before they hit the road, Daimler’s engineers have developed a robust testing system. But, optimizing the speed of their testing system for so many possible feature combinations is complex and time-consuming without a powerful algorithmic approach. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Daimler’s production lines.

![](https://storage.googleapis.com/kaggle-competitions/kaggle/6565/media/daimler-mercedes%20V02.jpg)

In this competition, **Daimler is challenging Kagglers to tackle the curse of dimensionality and reduce the time that cars spend on the test bench. Competitors will work with a dataset representing different permutations of Mercedes-Benz car features to predict the time it takes to pass testing. Winning algorithms will contribute to speedier testing, resulting in lower carbon dioxide emissions without reducing Daimler’s standards.**
'''

# %% [markdown]
'''
# Solution Approach

 - Step1
 - Step2
'''
# %% [markdown]
'''
**Author** : Sunil Yadav || yadav.sunil83@gmail.com || +91 96206 38383 ||
'''
# %% [markdown]
'''
# Solution
'''

# %% [markdown]
'''
# Lib Imports
'''
# %%
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
from enum import Enum, auto
%matplotlib inline

# %% [markdown]
'''
# Data load
'''
# %%
base_path = '../../data/'
train = pd.read_csv(f"{base_path}train.csv.zip", compression='zip')
test = pd.read_csv(f"{base_path}test.csv.zip", compression='zip')
# %%
print(train.shape, test.shape)
# %% [markdown]
'''
# Pandas settings
'''
# %%
pd.options.display.max_columns = None
pd.options.display.max_rows = 500
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3


# %% [markdown]
'''
# Data Overview
'''
# %%
print(train.shape, test.shape)
# %%
train.info(verbose=1)
# %%
train.sample(10)
# %%
(train.isnull().sum()/train.shape[0])*100
# %%
train.isnull().mean()*100
# %%
# Count of data types
train.dtypes.value_counts()
# %% [markdown]
'''
# EDA
'''
# %% [markdown]
'''
# EDA Utility Functions
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


# %%
id_cols = ["ID"]
target_col = 'y'
# %% [markdown]
'''
# Univariate
'''
# %% [markdown]
'''
# value counts
'''

# %%
unique_val_frame = print_count_of_uniques(train)
single_valued_cols = sorted(list(unique_val_frame[unique_val_frame["Column Unique Count"] == 1]["Column"]))
display(unique_val_frame.style.hide_index(), single_valued_cols)


# %%
str_cols = list(train.select_dtypes('object').columns)

binary_cols = [c for c in train.columns if ((c not in single_valued_cols) and (c not in id_cols) and (c not in str_cols) and c != target_col)]


# %%
print_value_count_percents(list(str_cols), train)
# %%
print_count_of_uniques(train[binary_cols])


# %% [markdown]
'''
## Drop unwanted columns
'''
# %%
# Drop unwanted columns
cols_to_drop = id_cols + single_valued_cols
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

# %%
print(train.shape, test.shape)
# %% [markdown]
'''
# distributions
'''
# %% [markdown]
'''
# Plotting numeric and categorical
'''
# %% [markdown]
'''
## Numerics
'''
# %%
#  plot target
plot_dist(train, [target_col], show_rug=False)

# %% [markdown]
'''
## categorical
'''

# %%
# Plot string valued columns frequency
plot_univariate_categorical_columns(list(str_cols), train, log_y=True, plot_limit=50)

# %%
for col in str_cols:
    plot_box(train, col, 'y')
# %%
# Some binary columns
plot_univariate_categorical_columns(['X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356', 'X357'], train, log_y=True)
# %% [markdown]
'''
# Bi-variate
'''
# %% [markdown]
'''
# Correlation
'''
# %%
train_corr = train.corr()
# %%
fig = px.imshow(train_corr)
fig.update_layout(width=1000, height=1000)
fig.show()


# %% [markdown]
'''
# Numeric-Numeric (Scatter plot)
'''
# %%
#  pair plot
fig = px.scatter_matrix(train[train_corr[target_col].abs().nlargest(16).index], width=1000, height=1000)
fig.show()
# %% [markdown]
'''
# Numeric-Categorical (Box and violin)
'''
# %%
#  get top 15 cols correlated with target column
top_15_cols = train_corr[target_col].abs().nlargest(16)[1:].index

for c in top_15_cols:
    plot_box(train, c, 'y')
# %% [markdown]
'''
# Categorical-Categorical (Cross Table) - How to do it?
'''
# %%
# TODO: Zero-One count compare plot
# %% [markdown]
'''
# Pre processing
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


# %%
train = replace_values_having_less_count(train, train.select_dtypes('object').columns)
# %%
print_value_count_percents(list(str_cols), train)

# %%
for col in str_cols:
    plot_box(train, col, 'y')
# %% [markdown]
'''
## Fix column dtypes - NA
'''

# %%
display(train.dtypes, test.dtypes)

# %% [markdown]
'''
# Null Handling - NA
'''
# %%
print(train.isna().sum().sum(),
      test.isna().sum().sum())
# %% [markdown]
'''
# Scaling - NA
'''
# %% [markdown]
'''
# Conversion of categorical (OHE or mean)
'''
# %%
encoded_train = train.copy()
encoded_test = test.copy()

# Could do get_dummies but test is separate sow we need to preserver encoders
col_encoder = {}
for col in str_cols:
    current_col_data = train[[col]]
    ohe = OneHotEncoder(handle_unknown='ignore').fit(current_col_data)
    transformed_train = ohe.transform(current_col_data).toarray()

    transformed_test = ohe.transform(test[[col]]).toarray()
    cols = [f"{col}_{c}" for c in ohe.categories_[0]]
    encoded_train = pd.concat([encoded_train, pd.DataFrame(np.array(transformed_train), columns=cols)],  axis=1)
    encoded_test = pd.concat([encoded_test, pd.DataFrame(np.array(transformed_test), columns=cols)], axis=1)
    encoded_train.drop(col, axis=1, inplace=True)
    encoded_test.drop(col, axis=1, inplace=True)

# %%
print(train.shape, encoded_train.shape, test.shape, encoded_test.shape)
# %% [markdown]
'''
# Outlier Treatment - NA
'''
# %% [markdown]
'''
# Single valued removal - Done
'''
# %% [markdown]
'''
# ID Removal - Done
'''
# %% [markdown]
'''
# Non important column removal - NA
'''
# %% [markdown]
'''
# Feature creation - NA
'''

# %% [markdown]
'''
# Dimensionality Reduction
'''
# %%
#  lets PCA with 90% information preservation
pca = PCA(0.9)
X = encoded_train.drop(target_col, axis=1)
y = encoded_train[target_col]
pca.fit(X)

encoded_train_dim_red = pca.transform(X)
encoded_test_dim_red = pca.transform(encoded_test)

# %%


def save_file(base_path, object, filename) -> None:
    with open(f"{base_path}{filename}.pkl", 'wb') as f:
        f.write(pickle.dumps(object))


save_file(base_path, encoded_train, 'encoded_train')
save_file(base_path, encoded_test, 'encoded_test')
save_file(base_path, encoded_train_dim_red, 'encoded_train_dim_red')
save_file(base_path, encoded_test_dim_red, 'encoded_test_dim_red')

# encoded_train.to_pickle(f"{base_path}encoded_train.pkl")
# encoded_test.to_pickle(f"{base_path}encoded_test.pkl")
# encoded_train_dim_red.to_pickle(f"{base_path}encoded_train_dim_red.pkl")
# encoded_test_dim_red.to_pickle(f"{base_path}encoded_test_dim_red.pkl")
# %%

# %%

# %%

# %%
