# %%
from typing import Tuple
import numpy as np
import pandas as pd
from math import sqrt
# %%
# %%
df = pd.read_csv("Online+Retail.csv", encoding='iso-8859-2')

# %%
df.head()
# %%
df["Country"].unique()
# %%
round(sqrt((7-23)**2 + (50-34)**2), 2)
# %%


def find_euclidean_distances(a: Tuple, b: Tuple, round_place: int = 2):
    if len(a) == len(b):
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - b[i])**2
        return round(sqrt(sum), round_place)
    else:
        raise ValueError("Points are of different dimensions")


# %%
find_euclidean_distances((7, 50), (23, 34))
# %%
print(find_euclidean_distances((7, 50), (12, 12)),
      find_euclidean_distances((23, 34), (12, 12)))
# %%
print(find_euclidean_distances((2, 3), (1, 2)), find_euclidean_distances((4, 5), (1, 2)),
      find_euclidean_distances((6, 2), (1, 2)))
# %%
