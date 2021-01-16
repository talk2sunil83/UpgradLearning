# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from sklearn import cluster

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

from IPython.display import display

# %%
df = pd.read_excel("Main.xlsx")
df.head()
# %%
df.columns

# %%
clustering_df = df[['Percentage Illiterate', 'Percentage Graduate & above']]
# %%
scaled_clustering_df = StandardScaler().fit_transform(clustering_df)
# %%
scaled_clustering_df = pd.DataFrame(scaled_clustering_df, columns=['PerIll', 'PGnAbv'])
scaled_clustering_df.head()
# %%

for l in ["single", "complete", 'average']:
    merging = linkage(scaled_clustering_df, method=l, metric='euclidean')
    dendrogram(merging)
    plt.show()

# %%
df = pd.read_excel("Book1.xlsx")
df.head()
# %%
cri = pd.read_csv("Cricket.csv", encoding='iso-8859-1')

# %%
cri.head()
# %%
cluster_df = cri[['Ave', 'SR']]
cluster_df.head()
# %%
cluster_df.info()

# %%
scaled_cluster_df = StandardScaler().fit_transform(cluster_df)
scaled_cluster_df[:5]
# %%
merging = linkage(scaled_cluster_df, method='complete', metric='euclidean')
dendrogram(merging)
plt.show()

# %%
cluster_labels = cut_tree(merging, n_clusters=4).reshape(-1, )
cluster_labels
# %%
cri['label'] = cluster_labels
# %%
cri
# %%
cri[cri['Player'].str.startswith('V Kohli')]['label']
# %%
cri[cri['label'] == 3].sort_values(by='Player')
# %%
for l in cri['label'].unique():
    display(cri[cri['label'] == l].sort_values(by='Player'))
# %%
