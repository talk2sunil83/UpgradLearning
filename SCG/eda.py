# %%
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
# %%
df = pd.read_csv("session3_operations.csv")

# %%
df.shape
# %%
df.isnull().sum()

# %%
round(df.isnull().sum()/df.shape[0], 2)
# %%
df.info()

# %%
df.describe()
# %%
# profile = ProfileReport(df, title='Data Profiling Report', minimal=True)
# profile.to_file("report.html")
# %%
# profile = ProfileReport(df, title='Data Profiling Report', explorative=True)
# profile.to_file("report_e.html")

# %%
# %%
# profile = ProfileReport(df, title='Data Profiling Report')
# profile.to_file("report_f.html")

# %%
