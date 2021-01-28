# %%
from math import sqrt
import numpy as np
from numpy.lib.npyio import save
from numpy.lib.shape_base import split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
%matplotlib inline
# %%
df = pd.read_csv("Cricket.csv", encoding='iso-8859-1')
# %%
df
# %%
df.columns
# %%
df.isnull().sum()
# %%
df.info()
# %%
df.describe().T
# %%
cluster_df = df[['Ave', 'SR']]
# cluster_df = df[['Span', 'Mat', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR',
#                  '100', '50', '0']]
# %%


def calc_span(v: str):
    parts = v.split('-')
    return int(parts[1]) - int(parts[0])


# cluster_df['Duration'] = cluster_df['Span'].apply(calc_span)
# %%
cluster_df
# %%
# cluster_df.drop("Span", axis=1, inplace=True)
# %%
# cluster_df['HS'] = cluster_df['HS'].apply(lambda x: int(x.replace("*", "")))
# %%
cluster_df.info()
# %%

scaled_cluster_df = StandardScaler().fit_transform(cluster_df)
scaled_cluster_df
# %%
km = KMeans(n_clusters=4, random_state=100)
km.fit(scaled_cluster_df)
# %%
km.labels_
# %%
df["ClusterIndex"] = km.labels_
# %%
df
# %%
df[df['Player'].str.startswith('V Kohli')]['ClusterIndex']
# %%
df[df['ClusterIndex'] == 2].sort_values(by='Player')
# %%
cluster_df.max()
# %%
cluster_df.min()
# %%
n = 15


def is_prime(num: int) -> bool:
    is_prime_number = True
    for i in range(2, num):
        if num % i == 0:
            is_prime_number = False
            break
    return is_prime_number


for i in range(2, n+1):
    if is_prime(i):
        print(i)
# %%
string = "Hello"
# string = "HelloUpgrad"
prevChar = ""
curr_longest = ""
longest = ""

# string = string.lower()

for char in string:
    if (prevChar.lower() <= char.lower()):
        curr_longest += char
        if len(curr_longest) > len(longest):
            longest = curr_longest
    else:
        curr_longest = char
    prevChar = char
print(longest)
# %%
n = 4
total_star = (2*n - 1)
for i in range(n):
    print("#"*i + (total_star - 2*i)*"*"+"#"*i)

# %%
n = 4
total_star = (2*n - 1)
i = 1
while n >= 0:
    k = n-i
    # print("#"*i + (total_star - 2*(i+1))*"*"+"#"*i)
    print(k)
    n -= 1
    i += 1

# %%
n = 3
total_star = (2*n - 1)
for i in range(n-1, -1, -1):
    print("#"*i + (total_star - 2*i)*"*"+"#"*i)
# %%
# input_list = [2, 1, 3, 4, 1, 5, 6, 1, 7]
input_list = [2, 1, 3, 4, 1, 5, 6, 1, 7]
# input_list = [2, 1, 2]

minimas = []
for i in range(1, (len(input_list)-1)):
    print(input_list[(i-1)], input_list[(i)], input_list[(i + 1)])
    if (input_list[(i-1)] > input_list[i]) and (input_list[(i-1)] > input_list[i]):
        minimas.append(i)

minimas = [int(x) for x in minimas]
print(minimas)

# %%
