# %%
import graphviz
import pydotplus
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# %%
df = pd.read_csv("Delhi+Delights+Data.csv")
# %%
df.head()
# %%
df.columns = ["a1", "a2", "y"]

# %%
y = df["y"]
X = df[["a1", "a2"]]
# y = y.map(lambda x: 1 if x.lower() == "yes" else 0)
# %%
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, criterion='gini')

# %%
dt.fit(X, y)

# %%

dot_data = StringIO()

export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
# %%
df.head(30)

# %%
print(df[df["a1"] < 3]["y"].value_counts())
print(df[df["a1"] >= 3]["y"].value_counts())

# %%
df[df["a1"] < 3]
# %%
k =df[df["a1"] < 3]
k[k["a1"] ==1.5]
# %%
k[k["a1"] ==2.5]
# %%
