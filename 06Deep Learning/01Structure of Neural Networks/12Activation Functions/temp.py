# %%
from math import exp
from scipy.special import softmax, expit as sigmoid
import numpy as np

# %%
w = np.array([2, -6, 3])
x = np.array([3, 2, 1])
b = -1
# %%
yh = x@(w.T) + b
sigmoid(yh)
# %%
