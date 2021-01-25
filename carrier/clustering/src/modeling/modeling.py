# %%
from typing import Sequence
import set_base_path
import time
import datetime

from IPython.display import display

import src.utils.modeling as mu

import pandas as pd
import numpy as np
from enum import Enum, auto
import warnings


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from src.constants import *
from dataclasses import dataclass
import glob
import json

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
### read data
'''
# %%
JSON_FILE_PATH = Path(__file__).resolve().parents[5] / 'UpgradLearning_data'/'covid'/'OpenResearch'

# %%
metadata = pd.read_csv(INTERIM_DATA_PATH/'metadata.zip')

# %%
all_json = glob.glob(f'{JSON_FILE_PATH}/**/*.json', recursive=True)
len(all_json)

# %%


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Paper:
    file_paper_id: str
    file_abstract: str
    file_body_text: str
    file_title: str
    file_authors: Sequence[str]
# %%


paper_ids = []
abstracts = []
body_texts = []
titles = []
authors_lst = []


def fill_entry_from_file(file_path, debug=False):
    with open(file_path) as file:
        file_json_data = json.load(file)
        paper_id = file_json_data.get('paper_id', None)
        abstract = file_json_data.get('abstract', None)
        body_text = file_json_data.get('body_text', None)
        metadata = file_json_data.get('metadata', None)

        abstract = " ".join([para.get('text') for para in abstract]) if abstract and len(abstract) > 0 else "NOT_PROVIDED"
        body_text = " ".join([para.get('text') for para in body_text]) if body_text and len(body_text) > 0 else "NOT_PROVIDED"
        title = None
        authors = None
        if metadata:
            title = metadata.get('title', None)
            authors = metadata.get('authors', None)
            authors = " ".join([para.get('text') for para in authors]) if authors and len(authors) > 0 else "NOT_PROVIDED"
        if debug:
            print(f"""
                  paper_id :{paper_id},
                  abstract: {abstract},
                  body_text: {body_text},
                  title: {title},
                  authors: {authors}
                  """)
        paper_ids.append(paper_id)
        abstracts.append(abstract)
        body_texts.append(body_text)
        titles.append(title)
        authors_lst.append(authors)


# fill_entry_from_file(all_json[0], True)
# %%
for file_path in all_json:
    fill_entry_from_file(file_path)

# %%
files_info_dataframe = pd.DataFrame({
    "sha": paper_ids,
    "file_abstract": abstracts,
    "file_body_text": body_texts,
    "file_title": titles,
    "file_authors": authors_lst,
})

# %%
merged_db = files_info_dataframe.merge(metadata, how="left", on='sha')
# %%
# merged_db.to_csv(INTERIM_DATA_PATH/"merged.zip", index=False, compression='zip')

# %%
merged_db_with_features = mu.__add_features_helper__(merged_db)
# %%

merged_db_with_features.to_csv(INTERIM_DATA_PATH/"merged.zip", index=False, compression='zip')

# %%
# %%
merged_db = pd.read_csv(INTERIM_DATA_PATH/"merged.zip")
# %%
merged_db.info()

# %%
merged_db['abstract'].describe(include='all')
# %%
merged_db['file_abstract'].describe(include='all')

# %%
