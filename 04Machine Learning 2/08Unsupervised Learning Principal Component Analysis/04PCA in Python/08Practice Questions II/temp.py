# %%
from sklearn.decomposition import PCA
import pandas as pd
# %%
df = pd.read_csv("Ratings.csv")
# %%
pca = PCA()
pca.fit(df)
# %%
pca.explained_variance_ratio_
# %%
