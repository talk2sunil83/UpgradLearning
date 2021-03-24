# %%
from math import exp
from scipy.special import softmax, expit as sigmoid
import numpy as np

# %%
w0 = np.array([1, 1, -1])
w1 = np.array([2, 0, -1])
w2 = np.array([1, 2, 2])
xp = np.array([2, 1, 1])
W = np.array([[1, 1, -1], [2, 0, -1], [1, 2, 2]])


# %%
W.shape

# %%
w0.shape
# %%
exp_values = []
for w in W:
    exp_values.append(exp(np.dot(w, xp)))
total = sum(exp_values)
total
# %%
np.around(np.array(exp_values)/total, 3)
# %%
