# %%
import pandas as pd
import numpy as np
from enum import Enum, auto
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from IPython.display import display
from src.utils.common import parallelize_dataframe

warnings.filterwarnings('ignore')


# %%

def __add_features_helper__(df: pd.DataFrame) -> pd.DataFrame:
    df['abstract_word_count'] = df['file_abstract'].apply(lambda x: len(str(x).strip().split()))  # word count in abstract
    df['body_word_count'] = df['file_body_text'].apply(lambda x: len(str(x).strip().split()))  # word count in body
    df['body_unique_words'] = df['file_body_text'].apply(lambda x: len(set(str(x).split())))
    return df


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    return parallelize_dataframe(dataframe, __add_features_helper__)
