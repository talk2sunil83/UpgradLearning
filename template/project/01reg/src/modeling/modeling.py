# %%
import time
import datetime
from scipy.stats import uniform
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split

#  Moldeling Libraries
from sklearn.linear_model import LinearRegression
# Scoring Libraries
from sklearn.metrics import r2_score

from IPython.display import display
import xgboost as xgb
from traitlets.traitlets import Any, Dict

from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from enum import Enum, auto
import warnings


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

warnings.filterwarnings('ignore')
%matplotlib inline
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
# Modeling Utilities
'''
# %%


class DataMode(Enum):
    ENCODED = auto()
    ENCODED_DIM_RED = auto()


def split_data(X, y=None, test_fraction: float = 0.2, shuffle: bool = True, stratify=None):
    return train_test_split(X, y, test_size=test_fraction, random_state=42, shuffle=shuffle, stratify=stratify) if y is not None else train_test_split(X, test_size=test_fraction, random_state=42, shuffle=shuffle, stratify=stratify)


def plot_error_patterns(error_data, **kwargs) -> None:
    pass


def print_feature_importance():
    pass


def print_equation():
    pass

# %%


linear = 'algoname'
model_obj = 'model'
params_dict = 'params'
models = {
    linear: {model_obj: LinearRegression(), params_dict: dict()}
}

# %% [markdown]
'''
### read data
'''

# %%
base_path = '../../data/'
encoded_train = pd.read_pickle(f"{base_path}encoded_train.pkl")
encoded_train_dim_red = pd.read_pickle(f"{base_path}encoded_train_dim_red.pkl")

target_col_name = 'y'
# %% [markdown]
'''
# Train test splitResidual Analysis
'''
# %%
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = split_data(encoded_train.drop(target_col_name, axis=1), encoded_train[[target_col_name]])
X_train_dim_red, X_test_dim_red, y_train_dim_red, y_test_dim_red = split_data(encoded_train_dim_red, encoded_train[[target_col_name]])
# %% [markdown]
'''
# Class Balancing - NA
'''

# %% [markdown]
'''
# Create model
'''
# %%


def prepare_error_data_and_plot(model, X, actual_y: pd.DataFrame, pred_y, mode, model_name, plot: bool = False) -> pd.DataFrame:
    data_mode_str = "without PCA" if mode == DataMode.ENCODED else "with PCA"
    error_df = actual_y.copy()
    error_df["Pred"] = pred_y
    # error_df["Res"] = error_df["y"] - error_df["Pred"]
    # error_df.columns = ["Actual", "Predicted", "Residual"]

    if plot:
        print(f"Residual Analysis graphs for \"Train Data\" {model_name} {data_mode_str}")
        plot_error_patterns(error_df)
    return error_df
# %%


show_graphs = False
all_model_metrics = []
for model, model_config in models.items():
    print(f"Starting for {model.title()}")
    # With and without Dimensionality Reduction
    for mode in [DataMode.ENCODED, DataMode.ENCODED_DIM_RED]:
        X_train, X_test, y_train, y_test = (X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded) if mode == DataMode.ENCODED else (
            X_train_dim_red, X_test_dim_red, y_train_dim_red, y_test_dim_red)
        data_mode_str = "without PCA" if mode == DataMode.ENCODED else "with PCA"
        # Hyper- Parameter tuning
        ra_s_cv = RandomizedSearchCV(model_config.get(model_obj), model_config.get(params_dict), random_state=0, n_jobs=-1, cv=3, verbose=3, return_train_score=True)

        start_time = time.perf_counter()
        ra_s_cv.fit(X_train, y_train)
        end_time = time.perf_counter()

        train_pred = ra_s_cv.predict(X_train)
        test_pred = ra_s_cv.predict(X_test)

        print("-"*50)
        print(f"Best Estimator for {model} {data_mode_str} is {ra_s_cv.best_estimator_}\n")
        print(f"Best Params for {model} {data_mode_str} are {ra_s_cv.best_params_}\n")
        print(f"Cross validation Results for {model} {data_mode_str}\n")
        display(pd.DataFrame(ra_s_cv.cv_results_))
        print("-"*50)

        # Plot evaluation matrix
        prepare_error_data_and_plot(ra_s_cv, X_train, y_train, train_pred, mode, model, show_graphs)
        prepare_error_data_and_plot(ra_s_cv, X_test, y_test, test_pred, mode, model, show_graphs)
        # Record performance
        all_model_metrics.append([f"{model.title()} {data_mode_str if mode == DataMode.ENCODED_DIM_RED else ''}", ra_s_cv.best_score_, ra_s_cv.score(X_train, y_train),
                                  ra_s_cv.score(X_test, y_test.values),
                                  #   r2_score(y_train, train_pred), r2_score(y_test, test_pred), # Metrics on train and test
                                  end_time-start_time])
        print("="*50, "\n")
# %%

perf_frame = pd.DataFrame(all_model_metrics, columns=["Algo", "Best Training Score (CV)", "Train Score", "Test Score",
                                                      #   "Train R2", "Test R2", # Col name for metrics on train and test
                                                      "Time Taken(Sec)"])

perf_frame["R2Diff"] = perf_frame["Train R2"] - perf_frame["Test R2"]
perf_frame.sort_values(by=["R2Diff", "Test R2", "Train R2"], ascending=[True, False, False]).style.format({
    "Best Training Score (CV)": "{:.2f}",
    "Train Score": "{:.2f}",
    "Test Score": "{:.2f}",
    # "Train R2": "{:.2f}",
    # "Test R2": "{:.2f}"
}).hide_index()
# %% [markdown]
'''
# Best Model Selection
'''

# %% [markdown]
'''
# Tune the best model
'''

# %% [markdown]
'''
# Conclusion
'''
