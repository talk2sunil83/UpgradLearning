# %%
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import random
import numpy as np
import plotly.graph_objects as go
# %%
print(plotly.__version__)
# %%
# Jupyter setup
init_notebook_mode(connected=True)
# %%
data = [{"x": list(range(6)), "y": [random.randint(1, 10) for _ in range(6)]}]
iplot(data)

# %%
x = np.random.randn(2000)
y = np.random.randn(2000)


# iplot([go.Histogram2dContour(x=x, y=y, coutours=dict(coloring='heatmap')),
#     go.Scatter(x=x, y=y, mode='markers', marker=dict(color='white', size=3,opecity=0.3))], show_link=False)
# %%
# %%
N = 2000
# %%
example_x = np.random.rand(N)
example_y = np.random.rand(N)
# %%
trace = go.Scatter(x=example_x, y=example_y, mode='markers')

# %%
data = [trace]
iplot(data)

# %%
# breakdown each category
groups = ['Rent', 'Food', 'Bills', 'Miscellaneous']
# create amount
amount = [1000, 518, 331, 277]
# style
colors = ['#d32c58', '#f9b1ee', '#b7f9b1', '#b1f5f9']
