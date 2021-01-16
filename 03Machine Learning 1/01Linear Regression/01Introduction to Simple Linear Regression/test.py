# %%
import numpy as np
import pandas as pd

# %%
pd.DataFrame({
    "x": range(1, 6),
    "y": np.linspace(0.5, 2.5, 5)
}).corr()
# %%
pd.DataFrame({
    "x": np.linspace(1, 5, 5),
    "y": np.linspace(1, 5, 5)
}).corr()
# %%
df = pd.DataFrame({
    "x": [1, 3, 6, 10, 15],
    "y": [89, 85, 79, 73, 64]
})

# %%
df.corr()
# %% [markdown]
'''

'''
