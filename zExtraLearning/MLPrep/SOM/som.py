# %%
from pandas.core.frame import DataFrame
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pickle
# %%
claims_data = pickle.load(open("scaled_data.pkl", 'rb'))
# # %%

# # %%
svd = TruncatedSVD(n_components=100)
transformed_data = svd.fit_transform(claims_data)

# %%
type(transformed_data)
# %%
pickle.dump(transformed_data, open("transformed_data.pkl", 'wb'))
# %%
pickle.dump(svd, open("svd.pkl", 'wb'))
# %%
# 100d Data
transformed_data = pickle.load(open("transformed_data.pkl", 'rb'))
svd = pickle.load(open("svd.pkl", 'rb'))
# # %%
# transformed_data[:1]
# # %%
# print(svd.explained_variance_ratio_.sum())
# # %%
# print(svd.singular_values_)
# # %%
# svd.inverse_transform(transformed_data[:1])
# %%
X_embedded = TruncatedSVD(n_components=3).fit_transform(transformed_data)
# # %%
# # %%
# X_embedded
# # %%

# X_embedded.shape
# # %%
# frame = DataFrame(X_embedded, columns=['0', '1', '2'])
# frame.columns

# # %%
# fig = px.scatter_3d(frame, x='0', y='1', z='2', opacity=0.7, width=2000, height=2000)
# fig.show()

# %%
