from re import T
import set_base_path
import numpy as np
import pandas as pd
from IPython.display import display
import plotly.figure_factory as ff
from enum import Enum, auto
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


def print_null_percents(frame: pd.DataFrame, full: bool = False, display_cols=True) -> pd.Series:
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
    print(f"Columns count with null: {len(null_counts)}")
    return null_counts


class GraphType(Enum):
    """Graph Type Enum

    Args:
        Enum ([type]): Built-in Enum Class
    """
    BAR = auto()
    LINE = auto()
    DIST = auto()


def __plot_univariate_series__(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        graph_type: GraphType = None,
        showlegend: bool = False,
        log_x: bool = False,
        log_y: bool = False,
        interactive: bool = False,
        x_rotation: int = None,
        y_rotation: int = None,
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

    if interactive:
        fig = None
        if graph_type is None or graph_type == GraphType.BAR:
            fig = px.bar(x=series.index, y=series, color=series.index,
                         title=title, labels=labels, log_x=log_x, log_y=log_y, **kwargs)

        if graph_type == GraphType.LINE:
            px.scatter(x=series.index, y=series, title=title, labels=labels, color=series.index, **kwargs)

        fig.update_layout(showlegend=showlegend)
        fig.show()
    else:
        plt.figure(figsize=(12, 10))
        ax = None
        if graph_type is None or graph_type == GraphType.BAR:
            ax = sns.barplot(x=series.index, y=series, palette="deep", **kwargs)

        if graph_type == GraphType.LINE:
            ax = sns.lineplot(x=series.index, y=series, **kwargs)

        ax.set(xlabel=xlabel, ylabel=ylabel, **kwargs)

        if x_rotation:
            plt.xticks(rotation=x_rotation)
        if y_rotation:
            plt.yticks(rotation=y_rotation)
        plt.show()


def get_univariate_cat_plot_strs(value: str, **kwargs) -> Tuple[str, str, str]:
    """Creates graph title, x-axis text and y-axis text for given value

    Args:
        value (str): column name

    Returns:
        Tuple[str, str, str]: title, x-axis text and y-axis text
    """
    full_name = value.replace("_", " ").replace("-", " ").replace(".", " ").title()  # TODO: write logic to make name
    if len(full_name) > 30:
        full_name = value
    count_str = full_name + ' Count' + " - Log Scale" if kwargs.get("log_y") else ""
    return count_str + ' Plot', full_name, count_str


def __plot_cat_data__(col_name: str, value_counts_ser: pd.Series, x_rotation=None, y_rotation=None, interactive=False, **kwargs):
    """Plots the value count series

    Args:
        c ([str]): column name
        value_counts_ser ([pd.Series]): value counts series
    """
    t, xl, yl = get_univariate_cat_plot_strs(col_name, **kwargs)
    __plot_univariate_series__(value_counts_ser, t, xl, yl, x_rotation=x_rotation, y_rotation=y_rotation, interactive=interactive, **kwargs)


def plot_univariate_categorical_columns(dataframe: pd.DataFrame, plot_limit: int = 30, print_value_counts=False, x_rotation=None, y_rotation=None, interactive=False, ** kwargs) -> None:
    """plots categorical variable bars

    Args:
        dataframe (pd.DataFrame): data frame with all categorical columns
        plot_limit (int, optional): plot if category count is less than. Defaults to 30.
        print_value_counts (bool, optional): print value counts or not. Defaults to False.
        x_rotation ([type], optional): x-axis text rotation angle (in degrees with non-interactive module). Defaults to None.
        y_rotation ([type], optional): y-axis text rotation angle (in degrees with non-interactive module). Defaults to None.
        interactive (bool, optional): if plot to be interactive (slow and make notebook more in size). Defaults to False.
    """
    for c in dataframe.columns:
        value_counts_ser = dataframe[c].value_counts().sort_values(ascending=False)
        if print_value_counts:
            print(value_counts_ser)
        cnt_len = len(value_counts_ser)
        if cnt_len > plot_limit:
            value_counts_ser = value_counts_ser[:plot_limit]
            print(f"Plotting only top(in decending order) {plot_limit} categories")
        __plot_cat_data__(c, value_counts_ser, x_rotation=x_rotation, y_rotation=y_rotation, interactive=interactive, ** kwargs)


def plot_dist(data_frame: pd.DataFrame, merge_all: bool = False, width=800, interactive: bool = False, **kwargs) -> None:
    cols_to_plot = data_frame.columns
    if interactive:
        if merge_all:
            fig = ff.create_distplot(hist_data=data_frame, group_labels=cols_to_plot, **kwargs)
            fig.update_layout(title_text=f"Dist plot for Numeric Columns", width=width)
            fig.show()
        else:
            for c in cols_to_plot:
                fig = ff.create_distplot(hist_data=[data_frame[c].values], group_labels=[c], **kwargs)
                fig.update_layout(title_text=f"Distribution plot for {c}", width=width)
                fig.show()
    else:
        if merge_all:
            sns.displot(data=data_frame, y=cols_to_plot, hue=cols_to_plot, **kwargs)
            plt.show()
        else:
            for c in cols_to_plot:
                sns.displot(data=data_frame, x=c, kind='kde', **kwargs)
                plt.show()


class TwoVarPlotType(Enum):
    BOX = auto()
    SCATTER = auto()


def __plot_two_features__(df: pd.DataFrame, x: str, y: str, plot_type: TwoVarPlotType, **kwargs):
    x_rotation = kwargs.get('x_rotation', 0)
    y_rotation = kwargs.get('y_rotation', 0)
    legend = kwargs.get('legend', None)
    _ = [kwargs.pop(key, None) for key in ['x_rotation', 'y_rotation', 'legend']]
    ax = None
    if plot_type == TwoVarPlotType.BOX:
        ax = sns.boxplot(data=df, x=x, y=y, hue=x, **kwargs)
    if plot_type == TwoVarPlotType.SCATTER:
        ax = sns.scatterplot(data=df, x=x, y=y, **kwargs)
    if ax is not None:
        plt.xticks(rotation=int(x_rotation))
        plt.yticks(rotation=int(y_rotation))
        if legend is not None and not legend:
            lgnd = ax.get_legend()
            if lgnd is not None:
                lgnd.remove()
        plt.show()


def plot_box(df: pd.DataFrame, x: str, y: str, interactive: bool = False, **kwargs) -> None:
    if interactive:
        fig = px.box(df, x=x, y=y, color=x, **kwargs)
        fig.show()
    else:
        __plot_two_features__(df, x, y, TwoVarPlotType.BOX, **kwargs)


def __getdtype__(col_data: pd.Series):
    str_dtype = str(col_data.dtype)
    if str_dtype in 'iufc' or col_data.dtype in [np.int64, np.float64]:
        return 'num'
    elif str_dtype in 'OSUb' or col_data.dtype in ['object']:
        return 'cat'
    elif str_dtype in 'mM':
        return 'date'
    else:
        return None

# REFACTOR: Make it consistent


def plot_two_variables(df, x, y, interactive: bool = False, **kwargs):
    if __getdtype__(df[x]) == 'num' and __getdtype__(df[y]) == 'num':
        if interactive:
            fig = px.scatter(df, x=x, y=y, trendline="ols", **kwargs)
            fig.show()
        else:
            __plot_two_features__(df, x, y, TwoVarPlotType.BOX, **kwargs)

    elif (__getdtype__(df[x]) == 'cat' and __getdtype__(df[y]) == 'num'):
        plot_box(df, x, y, interactive, **kwargs)
    elif (__getdtype__(df[x]) == 'num' and __getdtype__(df[y]) == 'cat'):
        plot_box(df, y, x, interactive, **kwargs)


def __set_value_count_color__(value):
    return "background-color: rgba(217, 38, 38, 0.2)" if value < 100 else ''


def __set_value_count_percent_color__(value):
    return "background-color: rgba(221, 207, 155, 0.2)" if value <= 5. else ''


def print_value_count_percents(dataframe: pd.DataFrame) -> None:
    total_recs = dataframe.shape[0]
    # ret_values = {}
    for c in dataframe.columns:
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
            applymap(__set_value_count_percent_color__, subset=["Percent"]). \
            applymap(__set_value_count_color__, subset=["Value Counts"]). \
            hide_index()

        display(styled_df)

    # return ret_values


def count_of_uniques(dataframe: pd.DataFrame, display_res=False) -> pd.DataFrame:
    cols = dataframe.columns
    unique_values = []
    unique_len = []
    for c in cols:
        uniques = dataframe[c].unique()
        # unique_values.append(sorted(uniques))
        unique_values.append(uniques)
        unique_len.append(len(uniques))

    frame = pd.DataFrame({
        "Column": cols,
        "Unique Values": unique_values,
        "Column Unique Count": unique_len})
    frame.sort_values(by=["Column Unique Count", "Column"], ascending=[False, True], inplace=True)
    if display_res:
        display(frame.style.hide_index())
    return frame


def get_data_frame_overview(df: pd.DataFrame, data_sample_size: int = 5) -> None:
    # Shape
    print(f"Shape:\n{df.shape}")
    print("-"*50)
    # DTypes
    print(f"\nDTypes:\n{df.dtypes}")
    print("-"*50)
    # Total Nulls
    print(f"\nTotal Nulls:\n{df.isnull().sum().sum()}")
    print("-"*50)
    # Nulls
    null_percent = round(df.isnull().mean()*100, 2)
    print(f"\nNulls Percentage:\n{null_percent[null_percent > 0].sort_values(ascending=False)}")
    print("-"*50)
    # Duplicate
    print(f"\nDuplicate Rows count:\n{len(df) - len(df.drop_duplicates())}")
    print("-"*50)
    # Sample
    print(f"\nSample:\n")
    display(df.sample(data_sample_size))
    print("-"*50)
    # Head
    print(f"\nHead:\n")
    display(df.head(data_sample_size))
    print("-"*50)
    # tail
    print(f"\nTail:\n")
    display(df.tail(data_sample_size))
    print("-"*50)
    # Describe
    print(f"\nDescribe:\n")
    display(df.describe(include='all', ).T)
    print("-"*50)
    # info
    print(f"\nInfo:\n")
    display(df.info(verbose=1))
    print("-"*50)
    # Count of data types
    print(f"\nCount of data types:\n")
    display(df.dtypes.value_counts())
    print("-"*50)
    # Column Names
    print(f"\nColumn Names:\n")
    display(df.columns)
