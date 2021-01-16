from enum import Enum, auto
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff


class DataMode(Enum):
    ENCODED = auto()
    ENCODED_DIM_RED = auto()


def split_data(X, y=None, test_fraction: float = 0.2, shuffle: bool = True, stratify=None):
    return train_test_split(X, y, test_size=test_fraction, random_state=42, shuffle=shuffle, stratify=stratify) if y is not None else train_test_split(X, test_size=test_fraction, random_state=42, shuffle=shuffle, stratify=stratify)


def plot_error_patterns(error_data, **kwargs) -> None:
    fig = ff.create_distplot(hist_data=[error_data["Residual"]], group_labels=["Residual"], **kwargs)
    fig.update_layout(title_text="Residuals Distribution")
    fig.show()

    sm.qqplot(error_data["Residual"], line='45', **kwargs)
    plt
    plt.show()

    fig = px.scatter(error_data, x="Actual", y="Predicted",  trendline="ols", **kwargs)
    fig.update_layout(title_text="Actual vs Predicted values")
    fig.show()

    fig = px.scatter(error_data, x="Predicted", y="Residual",  trendline="ols", **kwargs)
    fig.update_layout(title_text="Predicted values vs Residuals")
    fig.show()


def print_feature_importance():
    pass


def print_equation():
    pass
