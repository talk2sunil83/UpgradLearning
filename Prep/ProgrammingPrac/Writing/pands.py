# %%
import functools
import pandas as pd
import numpy as np
# %%

df = pd.DataFrame({"C1": [1, 2, 3, 4], "C2": [101, 202, 303, 404], "C3": [10, 52, -25, -40]})

# %%
C1 = "C1"
C2 = "C2"
C3 = "C3"

# %%
# if-then
df.loc[df[C1] >= 2, C2] = -10
df

# %%
df = pd.DataFrame({"C1": [1, 2, 3, 4], "C2": [101, 202, 303, 404], "C3": [10, 52, -25, -40]})
# %%
df.loc[df[C1] >= 2, [C2, C3]] = 999
df
# %%
df = pd.DataFrame({"C1": [1, 2, 3, 4], "C2": [101, 202, 303, 404], "C3": [10, 52, -25, -40]})
# %%
# Masking

df_mask = pd.DataFrame({C1: [True] * 4, C2: [False] * 4, C3: [True, False] * 2})
df.where(df_mask, -1000)
# %%
# if-then-else using Numpy where()
df["logic"] = np.where(df[C1] > 2, "high", "low")
df
# %%
df = pd.DataFrame({"C1": [1, 2, 3, 4], "C2": [101, 202, 303, 404], "C3": [10, 52, -25, -40]})
# %%

# Splitting
df[df.C1 <= 2]
# %%
df[df.C1 > 2]
# %%

# Building criteria
# and

df.loc[(df[C2] < 300) & (df[C3] >= -20), C1]
# %%
df = pd.DataFrame({"C1": [1, 2, 3, 4], "C2": [101, 202, 303, 404], "C3": [10, 52, -25, -40]})

# %%

# OR
df.loc[(df[C2] > 300) | (df[C3] >= -20), C1]

# %%
df = pd.DataFrame({"C1": [1, 2, 3, 4], "C2": [101, 202, 303, 404], "C3": [10, 52, -25, -40]})

# %%
Crt1 = df.C1 <= 2.2
Crt2 = df.C2 == 101.
Crt3 = df.C3 > -20.

AllCrit = Crt1 & Crt2 & Crt3
# %%
df[AllCrit]
# %%
CritList = [Crt1, Crt2, Crt3]
AllCrit = functools.reduce(lambda x, y: x & y, CritList)
df[AllCrit]

# %%
