

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from numpy import sqrt, sin, cos, pi
from scipy.integrate.quadpack import quad, dblquad, nquad
from scipy import special, integrate
import numpy as np
import scipy.special as special
# %%

(95 - 68)/2
# %%
(99.7 - 95)/4
# %%
(95 + 1.175)/100
# %%
(100-(50+34))/100
# %%


def get_Z_score(X: float, m: float, s: float) -> float:
    return (X-m)/s


# %%
m = 35.
s = 5.
print(get_Z_score(44.8, m, s), get_Z_score(25.2, m, s))
# %%
0.975-.025
# %%


def normal_dist_cdf_with_z(z: float) -> float:
    def integrate_fun(x): return np.exp((-x**2)/2)
    v = quad(integrate_fun, -np.inf, z)[0]
    v = (1/(sqrt(2*pi)))*v
    return v


def normal_dist_cdf(X: float, m: float, s: float):
    z = get_Z_score(X, m, s)
    return normal_dist_cdf_with_z(z), z


normal_dist_cdf_with_z(1.65)

# %%
normal_dist_cdf_with_z(-1.65)
# %%
m = 510
s = 20
X = 550
normal_dist_cdf(X, m, s)
# %%
X = 450
1 - normal_dist_cdf(X, m, s)[0]

# %%
m = 505
s = 25
XL = 450
XU = 550
normal_dist_cdf(XU, m, s)[0] - normal_dist_cdf(XL, m, s)[0]
# %%
m = 0
s = 1000
X = 2330
1 - normal_dist_cdf(X, m, s)[0]
# %%
m = 0
s = 1000
X = 500
# normal_dist_cdf(X,m, s)
get_Z_score(X, m, s)
# %%
normal_dist_cdf_with_z(0.5) - normal_dist_cdf_with_z(-0.5)
# %%
normal_dist_cdf_with_z(1.65) - normal_dist_cdf_with_z(-1.65)
# %%
# %%
normal_dist_cdf_with_z(2.58) - normal_dist_cdf_with_z(-2.58)
# %%


def get_confidence_interval(sample_mean: float, z_score: float, sample_standard_deviation: float, sample_size: int) -> Tuple[float, float]:
    margin_of_error = z_score*sample_standard_deviation/sqrt(sample_size)
    return (sample_mean-margin_of_error, sample_mean+margin_of_error, margin_of_error)


# %%
m = 0.505
s = 0.2
z = 1.96
n = 10_000
get_confidence_interval(m, z, s, n)
# %%
z = 2.58
get_confidence_interval(m, z, s, n)
# %%
n = 100
m = 530
s = 100
z = 1.96
get_confidence_interval(m, z, s, n)
# %%
n = 100
m = 530
s = 100
z = 1.65
get_confidence_interval(m, z, s, n)
# %% [markdown]
'''
For BJP
'''
# %%
n = 100
m = 58
s = 49.6
z = 1.96
get_confidence_interval(m, z, s, n)
# %% [markdown]
'''
FOR INC
'''
# %%
n = 100
m = 42
s = 49.6
z = 1.96
get_confidence_interval(m, z, s, n)
# %%
%matplotlib inline
# %%
df = pd.read_csv("Inferential Statistics - Powai Flats Rent.csv")
df
# %%
df.isnull().sum()
# %%
n = df.shape[0]
m = df["Monthly Rent"].mean()
s = df["Monthly Rent"].std()
z = 1.65
get_confidence_interval(m, z, s, n)
# %%
z = 2.58
get_confidence_interval(m, z, s, n)

# %%
z = 1.96
get_confidence_interval(m, z, s, n)
# %%


1 - normal_dist_cdf(90, 100, 10)[0]
# %%
