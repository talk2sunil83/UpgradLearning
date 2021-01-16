# %%
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
# %%

df = dd.read_csv(
    '3498_1146042_compressed_Parking_Violations_Issued_-_Fiscal_Year_2017.csv.zip', compression='zip')
df
# %%
missing_values = df.isnull().sum()
missing_values
# %%
df.
