# %%
from numpy import random
import numpy as np
from typing import Sequence
from scipy.stats import binom, norm
# %%
seed = 0
n = 10
p = 0.5

# %%
# write your code here
#  n -> number of trials


def get_binomials(seed: int, n: int, p: float) -> Sequence[float]:
    return binom.rvs(n, p, size=10, random_state=seed)


s = get_binomials(seed, n, p)
print(s)
# %%
# write your code here

# %%

1 - norm.cdf(x=90, loc=100,scale=10)

# %%

