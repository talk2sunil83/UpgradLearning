# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as pe
import re

# %%
df =  pd.read_csv("tendulkar_ODI.csv")

# %%
df.head()

# %%
df.dtypes
# %%
df.Runs.unique()


# %%
def clean_num(run_str):
    run_str = "".join(re.findall('\d+',str(run_str)))
    return np.int32(run_str) if run_str!='' else None
# %%
df['Runs'] = df['Runs'].apply(clean_num)

# %%
df['4s'].unique()

# %%
df['4s'] = df['4s'].apply(clean_num)

# %%

df.dtypes

# %%
df.head()

# %%
df['Runs'].describe()

# %%
df['Runs'].plot.hist(bins=20)

# %%
df['4s'].plot.hist(bins=25)

# %%
