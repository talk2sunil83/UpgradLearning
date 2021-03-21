# %%
import numpy as np

# %%
a11 = np.array([[1, 2], [5, 6]])
a12 = np.array([[3], [7]])
a21 = np.array([[9, 12], [10, 11]])
a22 = np.array([[13], [12]])

b11 = np.array([[1, 2], [12, 9]])
b12 = np.array([[3], [5]])
b21 = np.array([[10, 11]])
b22 = np.array([[12]])
# %%
for m in [a11, a12, a21, a22, b11, b12, b21, b22]:
    print(m.shape)

# %%
a11@b11 + a12@b21
# %%
a11@b12 + a12@b22
# %%
A = np.array([[1, 2, 3], [5, 6, 7], [9, 12, 13], [10, 11, 12]])
B = np.array([[1, 2, 3], [12, 9, 5], [10, 11, 12]])
A@B
# %%
