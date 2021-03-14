# %%
import math
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid, logit
# %%
np.random.seed(0)
# %%


def generate_dataset(N_points):
    # 1 class
    radiuses = np.random.uniform(0, 0.5, size=N_points//2)
    angles = np.random.uniform(0, 2*math.pi, size=N_points//2)

    x_1 = np.multiply(radiuses, np.cos(angles)).reshape(N_points//2, 1)
    x_2 = np.multiply(radiuses, np.sin(angles)).reshape(N_points//2, 1)
    X_class_1 = np.concatenate((x_1, x_2), axis=1)
    Y_class_1 = np.full((N_points//2,), 1)

    # 0 class
    radiuses = np.random.uniform(0.6, 1, size=N_points//2)
    angles = np.random.uniform(0, 2*math.pi, size=N_points//2)

    x_1 = np.multiply(radiuses, np.cos(angles)).reshape(N_points//2, 1)
    x_2 = np.multiply(radiuses, np.sin(angles)).reshape(N_points//2, 1)
    X_class_0 = np.concatenate((x_1, x_2), axis=1)
    Y_class_0 = np.full((N_points//2,), 0)

    X = np.concatenate((X_class_1, X_class_0), axis=0)
    Y = np.concatenate((Y_class_1, Y_class_0), axis=0)
    return X, Y


N_points = 1000
X, Y = generate_dataset(N_points)

plt.scatter(X[:N_points//2, 0], X[:N_points//2, 1], color='red', label='class 1')
plt.scatter(X[N_points//2:, 0], X[N_points//2:, 1], color='blue', label='class 0')
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.show()
# %%
X.shape, Y.shape
# %%
weights = dict(
    W1=np.random.randn(3, 2),
    b1=np.zeros(3),
    W2=np.random.randn(3),
    b2=0
)
# weights = {'W1': np.array([[-0.1049797,  1.36741498],
#                            [-1.65534404,  0.15364446],
#                            [-1.58447356,  0.84445431]]),
#            'b1': np.array([0., 0., 0.]),
#            'W2': np.array([-1.21286782,  0.28376955, -0.28219588]),
#            'b2': 0}
initial_weights = copy.deepcopy(weights)

# %%


def forward_propagation(X, weights):
    Z1 = X@weights['W1'].T + weights['b1']
    H = sigmoid(Z1)

    Z2 = H@weights['W2']+weights['b2']
    Y = sigmoid(Z2)

    return Y, Z2, H, Z1


# %%
forward_propagation(X, weights)

# %%
