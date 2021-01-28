# %%
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# %%
# Other type of file could be used which contains tabular data
advertising = pd.read_csv("advertising.csv")
# Target column must be last to work below all cell's code correctly, If you don't have your target colum last then make necessary changes to below two lines of code
TV = 'TV'
Radio = "Radio"
Newspaper = 'Newspaper'
Sales = 'Sales'

X = advertising.iloc[:, :1]
y = advertising.iloc[:, -1]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100
)
# %%
X_train
# %%
model = LinearRegression(
    normalize=True, fit_intercept=True, n_jobs=-1).fit(X_train, y_train)
# %%
y_predicted = model.predict(X_test)
# %%
r2_score(y_predicted, y_test)
# %%
