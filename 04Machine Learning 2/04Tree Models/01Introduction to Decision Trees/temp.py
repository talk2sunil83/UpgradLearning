# %%
import graphviz
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


def ginix(x: int, y: int) -> float:
    total = float(x+y)
    return 2*(x/total)*(y/total) if total > 0 else 0.5


# %%
ginix(80, 100)
# %%
ginix(120, 60)
# %%
# %%
# %%
X = pd.DataFrame({
    "X1": [1, 0, 1, 0],
    "X2": [1, 1, 0, 0],
    "X3": [1, 0, 1, 1]
})

y = pd.Series([1, 0, 0, 1])
# %%
# %%
dt = DecisionTreeClassifier()
dt.fit(X, y)
# %%
# plotting tree with max_depth=3
dot_data = StringIO()

export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns,
                class_names=['Zero', "One"])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
# %%
help(DecisionTreeClassifier)
# %%
