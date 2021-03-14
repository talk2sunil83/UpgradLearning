# %%
import numpy as np

# %%
xs = [np.array([[0], [3]]), np.array([[5], [9]]), np.array([[-1], [-2]])]
ws = [np.array([[-1], [1]]), np.array([[-1], [1]]), np.array([[-1], [2]]), np.array([[-1], [2]])]
bs = np.array([-2, -5, -1, 2])
ys = np.array([1, 1, 1, -1])
# %%

yh = 0
for i, w in enumerate(ws):
    for x in xs:
        mul_res = np.dot(w.T, x)
        yh += mul_res
    print((yh+bs[i])*ys[i])
    yh = 0
# %%
xs = [np.array([0, 3]), np.array([5, 9]), np.array([-1, -2])]
ws = [np.array([-1, 1]), np.array([-1, 1]), np.array([-1, 2]), np.array([-1, 2])]
bs = np.array([-2, -5, -1, 2])
ys = np.array([1, 1, 1, -1])
yh = 0
for i, w in enumerate(ws):
    for x in xs:
        mul_res = np.dot(x, w.T)
        yh += mul_res
    print((yh+bs[i])*ys[i])
    yh = 0
# %%
xs = np.array([[0, 3], [5, 9], [-1, -2]])
ws = np.array([[-1, 1], [-1, 1], [-1, 2], [-1, 2]])
print(xs.shape, ws.shape)
mul_res = np.dot(xs, ws.T)+bs
print(mul_res.shape)
print(np.dot(mul_res, ys))
# %%
