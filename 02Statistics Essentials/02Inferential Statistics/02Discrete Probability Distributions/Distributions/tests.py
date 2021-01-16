# %%
from dists.utils import Utils as u
from dists.binary_dist import BinaryDist

# %%

b = BinaryDist(10, 2, 0.05)


# %%
round((u.nCr(10, 2) * (0.05**2) * (0.95**8))*100, 2)
# %%
b.pdf()

# %%
b.cdf()
# %%
b.expected_value()

# %%
BinaryDist(10, 4, 0.4).pdf(True)
# %%
BinaryDist(10, 4, 0.4).pdf()
# %%
BinaryDist(10, 2, 0.4).cdf()

# %%
BinaryDist(10, 2, 0.4).cdf(True)

# %%
