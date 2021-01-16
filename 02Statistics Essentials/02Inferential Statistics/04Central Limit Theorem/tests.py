# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
sam = np.array([121.92, 133.21, 141.34, 126.23, 175.74])
# %%
m = sam.mean()
m
# %%

# %%


def get_stats(data: np.ndarray, is_sample: bool = False):
    if data is not None:
        if not is_sample:
            return np.mean(data), np.std(data)
        else:
            m = np.mean(data)
            return m, np.sqrt((sum((m-data)**2)) / (data.shape[0]-1))

# %%


get_stats(sam, True)


# %%
%matplotlib inline
# %%
df = pd.read_csv("Inferntial Statistics - UpGrad Samples.csv")
df

# %%
df['Sample Mean'].mean()
# %%
get_stats(df['Sample Mean'].values, False)
# %%
data = df['Sample Mean'].values
m = 2.348
np.sqrt((sum((m-data)**2)) / 99)
np.sqrt((sum((m-data)**2)) / 98)
# %%

np.sqrt(((((1-0.42)**2)*42) + (((0-0.42)**2)*58))/99)
# %%
μX = 0.50
SE = 0.052

(0.052 / 0.50)**2
# %%
μX = 0.50
SE = 0.048
