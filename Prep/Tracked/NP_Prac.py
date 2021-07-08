# %%
from typing import Union
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
np.random.seed(0)
%matplotlib inline
seaborn.set()  # set plot style
# %%
x1: np.ndarray = np.random.randint(10, size=6)
x2: np.ndarray = np.random.randint(10, size=(3, 4))
x3: np.ndarray = np.random.randint(10, size=(3, 4, 5))

# %%
print(f"{x3.ndim=}, {x3.shape=}, {x3.size=}, {x3.dtype=}")
# %%
print(f"{x3.itemsize=}, {x3.nbytes}")

# %%
x1
# %%
x1[0]
# %%
x1[-1]
# %%
x1[-2]
# %%
x2
# %%
x2[0, 0]
# %%
x2[2, 0]

# %%
x2[2, -1]

# %%
x1[0] = 3.14159
x1
# %%
x = np.arange(10)
x

# %%
x[:5]
# %%
x[5:]

# %%
x[::-1]
# %%
x[5::-2]  # reversed every other from index 5

# %%
x2

# %%
x2[:2, :3]  # two rows, three columns

# %%
x2[:3, ::2]  # all rows, every other column

# %%
x2[::-1, ::-1]

# %%
mark = {False: ' -', True: ' Y'}


def print_table(ntypes):
    print('X ' + ' '.join(ntypes))
    for row in ntypes:
        print(row, end='')
        for col in ntypes:
            print(mark[np.can_cast(row, col)], end='')
        print()


print_table(np.typecodes['All'])
# %%

L = np.random.random(100)
sum(L)
# %%
big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)

# %%
min(big_array), max(big_array)

# %%
np.min(big_array), np.max(big_array)

# %%
%timeit min(big_array)
%timeit np.min(big_array)

# %%
print(big_array.min(), big_array.max(), big_array.sum())

# %%
M = np.random.random((3, 4))
print(M)

# %%
M.sum()

# %%
M.sum(axis=0), M.sum(axis=1)

# %%
M.min(axis=0)


# %%
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

# %%
print("Mean height:       ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height:    ", heights.min())
print("Maximum height:    ", heights.max())

# %%
print("25th percentile:   ", np.percentile(heights, 25))
print("Median:            ", np.median(heights))
print("75th percentile:   ", np.percentile(heights, 75))

# %%
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')
plt.show()
# %%
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

# %%
a + 5
# %%
M = np.ones((3, 3))
M

# %%
M + a

# %%
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]

print(a)
print(b)

# %%
# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# %%
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar()

# %%
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0  # 1/10mm -> inches
inches.shape

# %%
plt.hist(inches, 40)
plt.show()
# %%
print("Number days without rain:      ", np.sum(inches == 0))
print("Number days with rain:         ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches  :", np.sum((inches > 0) & (inches < 0.2)))

# %%
# construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2014 (inches):   ", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):  ", np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ", np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):", np.median(inches[rainy & ~summer]))

# %%
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)

# %%
#  O[N^2]


def selection_sort(num_array: Union[int, float]) -> Union[int, float]:
    for i in range(len(num_array)):
        swap_idx = i + np.argmin(num_array[i:])
        (num_array[i], num_array[swap_idx]) = (num_array[swap_idx], num_array[i])
    return num_array


x = np.array([2, 1, 4, 3, 5])
selection_sort(x)

# %%
# worst sorting algorithm : O[N×N!]


def bogosort(num_array: Union[int, float]) -> Union[int, float]:
    counter = 0
    while np.any(num_array[:-1] > num_array[1:]):
        np.random.shuffle(num_array)
        counter += 1
    print(f"{counter=}")
    return num_array


x = np.array([2, 1, 4, 3, 5])
bogosort(x)

# %%
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)

# %%
np.partition(X, 2, axis=1)

# %%
