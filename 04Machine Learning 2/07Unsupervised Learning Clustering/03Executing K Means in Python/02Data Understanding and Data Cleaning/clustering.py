# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import factorize
import seaborn as sns
%matplotlib inline
# %%
df = pd.read_excel("Online Retail.xlsx")

# %%
df.shape
# %%
df.info()
# %%
df.isnull().sum()
# %%
((df.isnull().sum()/df.shape[0])*100).round(2).sort_values(ascending=False).apply(lambda x: str(x)+"%")

# %%
df.dropna(inplace=True)
# %%
df.shape
# %%
((df.isnull().sum()/df.shape[0])*100).round(2).sort_values(ascending=False).apply(lambda x: str(x)+"%")
# %%
