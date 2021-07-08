# %%
import warnings
from matplotlib.pyplot import fill
import pandas as pd
import numpy as np
rng = np.random.RandomState(42)
warnings.filterwarnings('ignore')
# %%
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
df
# %%
np.sin(df * np.pi / 4)
# %%
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
# %%
population / area
# %%
area.index | population.index
# %%
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B
# %%
A.add(B, fill_value=0)
# %%
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list("AB"))
A

# %%
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
B
# %%
A + B
# %%
A.stack()
# %%
A.stack().mean()
# %%
A.mean()
# %%
A.mean().mean()
# %%
fill_value = A.mean().mean()
A.add(B, fill_value=fill_value)
# %%
