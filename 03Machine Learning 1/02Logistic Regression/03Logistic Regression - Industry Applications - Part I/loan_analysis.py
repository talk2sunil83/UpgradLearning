# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.arrays import categorical
import seaborn as sns
%matplotlib inline
# %%
df = pd.read_csv("loan.csv")
# %%
list(df.columns)
# %%
df['purpose'].unique()
# %%
