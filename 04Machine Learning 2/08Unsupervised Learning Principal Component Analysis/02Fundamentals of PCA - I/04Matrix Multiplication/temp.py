# %%
import numpy as np
M = np.array([[2, 1, 1], [3, 2, 1], [2, 1, 2]])
np.linalg.inv(M)
# %%
B1 = [[1, 1], [2, -1]]
B2 = [[1, 0], [0, 1]]
np.linalg.inv(B2)@B1
# %%
v1 = np.array([[8], [6]])
B1 = np.array([[1, 0], [0, 1]])
B2 = np.array([[2, -2], [1, 1]])

M = np.linalg.inv(B2)@B1
print(M)

v2 = M@v1
print(v2)

# %%
v1 = np.array([[25], [28]])
B1 = np.array([[1, 0], [0, 1]])
B2 = np.array([[1, 1], [-2, 1]])

M = np.linalg.inv(B2)@B1
print(M)

v2 = M@v1
print(v2)
# %%
v1 = np.array([[-3], [12]])
B1 = np.array([[1, 1], [-2, 1]])
B2 = np.array([[1, 0], [0, 1]])


M = np.linalg.inv(B2)@B1
print(M)

v2 = M@v1
print(v2)
# %%
v1 = np.array([[1], [1]])
B1 = np.array([[1, 2], [1, 1]])
B2 = np.array([[1, 0], [0, 1]])


M = np.linalg.inv(B2)@B1
print(M)

v2 = M@v1
print(v2)
# %%
v1 = np.array([[3], [2]])
B1 = np.array([[1, 0], [0, 1]])
B2 = np.array([[3, -3], [4, -5]])

M = np.linalg.inv(B2)@B1
print(M)

v2 = M@v1
print(v2)

# %%
