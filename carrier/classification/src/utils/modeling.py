# %%
from lazypredict.Supervised import Classification
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from enum import Enum, auto
import warnings

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from IPython.display import display
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')


# %%


class DataMode(Enum):
    ENCODED = auto()
    ENCODED_DIM_RED = auto()


def split_data(X, y=None, test_fraction: float = 0.2, shuffle: bool = True, stratify=None):
    return train_test_split(X, y, test_size=test_fraction, random_state=42, shuffle=shuffle, stratify=stratify) if y is not None else train_test_split(X, test_size=test_fraction, random_state=42, shuffle=shuffle, stratify=stratify)


def print_feature_importance():
    pass


def print_equation():
    pass


def prepare_error_data_and_plot(model, X, actual_y: pd.DataFrame, pred_y, mode, model_name, plot: bool = False) -> pd.DataFrame:
    data_mode_str = "without PCA" if mode == DataMode.ENCODED else "with PCA"
    error_df = actual_y.copy()
    error_df["Pred"] = pred_y
    display(confusion_matrix(actual_y, pred_y))

    if plot:
        print(f"Classification Analysis graphs for \"Train Data\" {model_name} {data_mode_str}")
        display(roc_curve(actual_y, pred_y))
    return error_df


def get_feature_importance(model, X_val, y_val):
    r = permutation_importance(model, X_val, y_val, n_repeats=3, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:

            print(f"{X_val.columns[i]:<30}\t"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
