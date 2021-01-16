# %%
from os import sep
import numpy as np
import pandas as pd
import plotly.express as px
# %%
# df = pd.read_csv("https://raw.githubusercontent.com/talk2sunil83/interviewpr/main/data.csv", sep="-")
df = pd.read_csv("total-electricity-consumption-us.csv")
df.head()
# %%
fig = px.scatter(df, x="Year", y="Consumption")
fig.show()
# %%
