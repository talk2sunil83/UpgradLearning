# %%
from scipy.special import expit as sigmoid
# %%
z2 = [2, 1, 3, -1]

# %%
h2 = sigmoid(z2)
h2
# %%
sp = h2*(1-h2)
sp
# %%
