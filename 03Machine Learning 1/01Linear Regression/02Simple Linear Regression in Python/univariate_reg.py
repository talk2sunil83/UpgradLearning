# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pl
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm
from IPython.display import display

# %% [markdown]
'''
<h3>Constants</h3>
'''
# %%
TV = 'TV'
Radio = "Radio"
Newspaper = 'Newspaper'
Sales = 'Sales'

# %%
advertising = pd.read_csv("advertising.csv")
advertising.head()
# %%
advertising.shape
# %%
advertising.info()
# %%
advertising.describe()
# %%
advertising.isnull().any()
# %%

# visualize data
sns.regplot(x=TV, y=Sales, data=advertising)
# %%
sns.regplot(x=Radio, y=Sales, data=advertising)

# %%
sns.regplot(x=Newspaper, y=Sales, data=advertising)

# %%
sns.pairplot(advertising, x_vars=[
             TV, Newspaper, Radio], y_vars=Sales, size=4, aspect=1, kind='scatter')
plt.show()

# %%
advertising.corr()

# %%
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot=True)
# %%
# create X and y

X = advertising[TV]
y = advertising[Sales]

# %%

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=100)
X_train
# %%
# training the model

X_train_sm = sm.add_constant(X_train)
X_train_sm.head()
# %%
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()

# %%
lr_model.params
# %%
lr_model.summary()
# %%
y_train_pred = lr_model.predict(X_train_sm)
y_train_pred
# %%
plt.scatter(X_train, y_train)
plt.plot(X_train, y_train_pred, 'r')
plt.show()

# %% [markdown]
'''
<h2>Residual Analysis</h2>
'''

# %%
# y_train, y_train_pred
res = y_train - y_train_pred
plt.figure()
sns.distplot(res)
plt.title("Residual Plot")
plt.show()
# %%
# look for patterns in Residual
plt.scatter(X_train, res)
plt.show()
# %% [markdown]
'''
<h2>Step 4: Predictions and evaluation on the test set</h2>
'''
# %%
# make test data
X_test_sm = sm.add_constant(X_test)
# predict on test data and eval data on r-squared and others

y_test_pred = lr_model.predict(X_test_sm)
# %%
y_test_pred.head()
# %%
r2_test = r2_score(y_test, y_test_pred)
r2_test
# %%
mean_squared_error(y_test, y_test_pred)
# %%
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()

# %%
#  Reshape to 140,1
X_train_lm = X_train.values.reshape(-1, 1)
X_test_lm = X_test.values.reshape(-1, 1)
X_test_lm.shape
# %%
lm = LinearRegression()
lm.fit(X_train_lm, y_train)
# %%
display(lm.coef_, lm.intercept_)
# %%
# make Predictions
y_train_pred = lm.predict(X_train_lm)
y_test_pred = lm.predict(X_test_lm)

# %%
plt.scatter(X_train, y_train)
plt.plot(X_train, y_train_pred, 'r')
plt.show()
# %%
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()
# %%
print(r2_score(y_train, y_train_pred))
print(r2_score(y_test, y_test_pred))
# %%
