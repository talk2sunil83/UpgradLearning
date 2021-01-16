# %%
from typing import Sequence, Union
import numpy as np
import scipy as sp
from scipy.special import expit

# %%


def get_sigmoid(x: float) -> float:  # this is same as scipy.special.expit
    return 1/(1 + np.exp(-x))


# %%
get_sigmoid(-2)
# %%
expit(-2)
# %%


# %%
def get_p(x):
    z_val = -13.5 + (0.06 * x)
    print(f"z_val : {z_val}")
    return expit(z_val)


print(get_p(220))
print(get_p(231.5))
print(get_p(243))
# %%
