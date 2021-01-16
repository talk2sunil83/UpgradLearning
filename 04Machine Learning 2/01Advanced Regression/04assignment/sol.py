# %% [markdown]
'''
<h1>Australian market analysis and model building for Surprise Housing</h1>
<hr/>
<h2>Business Objective(s)</h2>

**The company wants to know:**
  - Which variables are significant in predicting the price of a house, and
  - How well those variables describe the price of a house.

**Technical goal**
  - Determine the optimal value of lambda for ridge and lasso regression

<h3>Business Goal</h3>
    You are required to model the price of houses with the available independent variables. This model will then be used by the management to understand how exactly the prices vary with the variables. They can accordingly manipulate the strategy of the firm and concentrate on areas that will yield high returns. Further, the model will be a good way for management to understand the pricing dynamics of a new market.
'''

'''
<h2>Libraries Imports</h2>
'''
# %%
# %% [markdown]
'''
<h3>Notebook settings</h3>
'''
# %%
# render matplotlib graphs inline
import warnings
from IPython.display import display
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.shape_base import tile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.figure_factory as ff
import plotly.graph_objects as go
from enum import Enum, auto
from typing import Sequence, Tuple
from pandas_profiling.profile_report import  ProfileReport
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
%matplotlib inline
# import display function
# Suppress Warnings
warnings.filterwarnings('ignore')

# %% [markdown]
'''
<h3>Pandas settings</h3>
'''
# %%
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3

# %% [markdown]
'''
<h4>Load Data</h4>
'''

# %%
df = pd.read_csv("train.csv")
# %% [markdown]
'''
<h3>Data peak</h3>
'''
# %%
# head
df.head()

# %%
df.info()
# %%
df.describe().T

# %%
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99]).T

# %% [markdown]
'''
Many columns have blank values
'''
# %%
print(f"Total Observations :{df.shape[0]}\n Total Columns: {df.shape[1]}")
# %% [markdown]
'''
<h3>SalePrice Analysis</h3>
'''
# %%
# Let's see the distribution of Sale price
sns.distplot(df['SalePrice'])
plt.show()

# %% [markdown]
'''
<h3>It seems there are some outliers, we need to analyze</h3>
'''
# %%
df.drop('Id', axis=1, inplace=True)
print(df.shape)
# %%
# Column wise Null percentage
nullper = round((df.isna().sum()[df.isna().sum() > 0]/df.shape[0])*100, 2).sort_values(ascending=False)
nullper
# %% [markdown]
'''
**Total 4 columns have more than 80% blank values, so we'll drop them**
'''
# %%
df_dropped = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
print('Shape of the orginal df:', df.shape)
print('Shape after dropping:', df_dropped.shape)

# %% [markdown]
'''
<h3>Pandas profiling</h3>
'''
# %%
# profile_rptr =  ProfileReport(df, title="Australian market analysis", explorative=True)
# profile_rptr.to_widgets()

# %% [markdown]
'''
<h3>Dividing columns to investigate</h3>
'''
# %%
# Categorical columns
cat_cols = ["MSSubClass", "MSZoning",
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            "Utilities",
            "LotConfig",
            "LandSlope",
            "Neighborhood",
            "Condition1",
            "Condition2",
            "BldgType",
            "HouseStyle",
            "OverallQual",
            "OverallCond",
            "RoofStyle",
            "RoofMatl",
            "Exterior1st",
            "Exterior2nd",
            "MasVnrType",
            "ExterQual",
            "ExterCond",
            "Foundation",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "Heating",
            "HeatingQC",
            "CentralAir",
            "Electrical",
            "KitchenQual",
            "Functional",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "PavedDrive",
            "PoolQC",
            "Fence",
            "MiscFeature",
            "SaleType",
            "SaleCondition"]

# Need some extra investigation
explr_cols = ["BsmtFullBath",
              "BsmtHalfBath",
              "FullBath",
              "HalfBath",
              "Bedroom",
              "Kitchen",
              "TotRmsAbvGrd",
              "Fireplaces",
              "GarageCars"]

# We'll extract age from these
age_cols = ["YearBuilt",
            "YearRemodAdd",
            "GarageYrBlt",
            "YrSold"]

# Numerical Columns
num_cols = ["LotFrontage", "LotArea", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
            "LowQualFinSF", "GrLivArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "SalePrice"]


# %% [markdown]
'''
<h3>Lets plot all numeric columns</h3>
'''
# %%
# Distribution plot for numerical columns
for num_col in num_cols:
    sns.displot(df, x=num_col, kind='kde', rug=True).set(title=f'Distribution Plot for {num_col}', ylabel=f'{num_col} values distribution')
    plt.show()
# %% [markdown]
'''
**Most of the columns have most values around zero**
'''
# %% [markdown]
'''
<h3>Lets plot all categorical columns</h3>
'''
# %%
'''
<h2>Utility Functions</h2>
'''
# %%


class GraphType(Enum):
    """Graph Type Enum

    Args:
        Enum ([type]): Built-in Enum Class
    """
    BAR = auto()
    LINE = auto()


def plot_univariate_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        graph_type: GraphType = None,
        **kwargs) -> None:
    """Bar plots a interger series

    Args:
        series (pd.Series): series to be plotted
        title (str): graph title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        display_format (str, optional): number format. Defaults to '{0:,.0f}'.
        figsize ([type], optional): figure size. Defaults to None.
        show_count (bool, optional): show value at the top of bar. Defaults to True.
        graph_type (GraphType, optional): graph type
    """
    labels = {"x": xlabel, "y": ylabel}
    fig = None
    if graph_type is None or graph_type == GraphType.BAR:
        fig = px.bar(x=series.index, y=series, color=series.index,
                     title=title, labels=labels, **kwargs)
    if graph_type == GraphType.LINE:
        px.scatter(x=series.index, y=series, title=title, labels=labels, color=series.index,
                   **kwargs)
    fig.show()


def get_univariate_cat_plot_strs(value: str) -> Tuple[str, str, str]:
    """Creates graph title, x-axis text and y-axis text for given value

    Args:
        value (str): column name

    Returns:
        Tuple[str, str, str]: title, x-axis text and y-axis text
    """
    title_case = value.replace('_', '').title()
    count_str = title_case + ' Count'
    return count_str + ' Plot', title_case, count_str


def plot_univariate_categorical_columns(categorical_cols: Sequence[str], dataframe: pd.DataFrame, **kwargs) -> None:
    """plots categorical variable bars

    Args:
        categorical_cols (Sequence[str]): categorical columns
        dataframe (pd.DataFrame): DataFrame
    """
    for c in categorical_cols:
        value_counts_ser = dataframe[c].value_counts()
        cnt_len = len(value_counts_ser)
        if cnt_len < 16:
            t, xl, yl = get_univariate_cat_plot_strs(c)
            plot_univariate_series(value_counts_ser, t, xl, yl, **kwargs)


# %%
# plot_univariate_categorical_columns(cat_cols, df)
# %%
rec_count = df.shape[0]
# Categorial columns value counts
info_db = []
value_counts_dict = {}
for col in cat_cols:
    # print(f"Value Counts for '{col}'")
    val_counts = df[col].value_counts()
    # display(val_counts)
    value_counts_dict[col] = pd.DataFrame({
        "Value": val_counts.index,
        "Count": val_counts.values,
        "Percent": round((val_counts*100)/rec_count, 2).values})
    null_count = df[col].isnull().sum()
    nulls_per = round((null_count*100)/rec_count, 2)
    info_db.append((col, len(val_counts), null_count, nulls_per))
    # print(f"Value Counts: {len(val_counts)}, Null Count: {null_count}, Null Percentage: {nulls_per}")
    # print("-"*50)
# %%
# Column , value count, null counts and null percent
pd.DataFrame(info_db, columns=["Column", "Value Type Count", "Null Count", "Null Percentage"]).sort_values(by=["Value Type Count", "Null Count"], ascending=False)

# %%
# Value percents
for k, v in value_counts_dict.items():
    print(k)
    display(v)

# %% [markdown]
'''
**For each categorical column if share value is less than 5% then put in other category**
'''
# %%
# the below replacement_dict will be used in transformation of test set
replacement_dict = {}
for k, v in value_counts_dict.items():
    v["NewValue"] = v.apply(lambda r: r["Value"] if r["Percent"] > 5. else "Others", axis=1)
    val_dicts = {}
    for r in range(v.shape[0]):
        val_dicts[v.iloc[r]['Value']] = v.iloc[r]["NewValue"]

    replacement_dict[k] = val_dicts

# %%
# Replacing values
for k, v in replacement_dict.items():
    df[k] = df[k].replace(v)

# %%
df.sample(20)

# %% [markdown]
'''
<h3>Imputing categorical columns</h3>
'''
# %%
# Garage columns
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col].fillna('NO GARAGE', inplace=True)
df['GarageYrBlt'].fillna(0, inplace=True)

# %%
# Basement columns
for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtFinType1', 'BsmtCond']:
    df[col].fillna('NO BASEMENT', inplace=True)

# %% [markdown]
'''
<h3>columns null check</h3>
'''
# %%
round((df.isna().sum()[df.isna().sum() > 0]/df.shape[0])*100, 2).sort_values(ascending=False)
# %% [markdown]
'''
<h3>Lets Investigate columns wit more than 80% nulls</h3>
'''
# %%
df[['PoolArea', 'PoolQC']].sample(30, random_state=997)

# %%
df['PoolQC'].unique()

# %%
#  check if any PoolQC with None have some values
df[df['PoolQC'].isna()]['PoolArea'].sum()

# %% [markdown]
'''
**
 1. We can safely drop PoolQC, and 
 2. Alley has NA => 'No alley access', MiscFeature has NA=> None, value already so no point to impute it, same goes wit Fence and FireplaceQu
**
'''
# %%
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', "FireplaceQu"], axis='columns')
# %%
df.shape
# %%
round((df.isna().sum()[df.isna().sum() > 0]/df.shape[0])*100, 2).sort_values(ascending=False)

# %% [markdown]
'''
<h3>We'll drop "MoSold" as well and use "YrSold" </h3>
'''
# %%
df = df.drop(["MoSold"], axis='columns')

# %%
round((df.isna().sum()[df.isna().sum() > 0]/df.shape[0])*100, 2).sort_values(ascending=False)
# %%
# Converting year columns to age columns
for age_col in age_cols:
    df[f'{age_col}_Old'] = df[age_col].max()-df[age_col]
    df.drop(age_col, axis=1, inplace=True)

# %%
#  value imputation's
df['MasVnrType'].fillna('None', inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)
df['BsmtQual'].fillna('TA', inplace=True)
df['BsmtCond'].fillna('TA', inplace=True)
df['BsmtExposure'].fillna('No', inplace=True)
df['BsmtFinType1'].fillna('Unf', inplace=True)
df['BsmtFinType2'].fillna('Unf', inplace=True)
df['GarageType'].fillna('Attchd', inplace=True)
df['GarageYrBlt_Old'].fillna(-1, inplace=True)
df['GarageFinish'].fillna('Unf', inplace=True)
df['GarageQual'].fillna('TA', inplace=True)
df['GarageCond'].fillna('TA', inplace=True)
# %% [markdown]
'''
<h2>Outliers treatment for numeric columns</h2>
'''
# %%
df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99]).T
# %% [markdown]
'''
**It seems few columns have outliers**
'''
# %%

# Removing outliers


def remove_outlier_data(x):
    list = []
    for col in num_cols:
        Q1 = x[col].quantile(.25)
        Q3 = x[col].quantile(.99)
        IQR = Q3-Q1
        x = x[(x[col] >= (Q1-(1.5*IQR))) & (x[col] <= (Q3+(1.5*IQR)))]
    return x


df = remove_outlier_data(df)

# %%
#  pairplot
sns.pairplot(df[num_cols])
plt.show()

# %%
# Correlation plot
plt.figure(figsize=(16, 16))
sns.heatmap(df[num_cols].corr(), annot=True)
plt.show()

# %% [markdown]
'''
<h2>Lets Get dummies for categorical columns</h2>
'''
# %%
# Need to do because we have dropped many categorical columns

cat_cols = list(set(cat_cols).intersection(set(df.columns)))

# %%
dummies = pd.get_dummies(df[cat_cols], drop_first=True)
df = pd.concat([df, dummies], axis=1)
df = df.drop(cat_cols, axis=1)

# %%
list(df.columns)
# %%
df.shape
# %%
df.sample(10)
# %%
# %% [markdown]
'''
<h2>Split the data</h2>
'''
# %%
df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, random_state=42)

# %%
num_col = list(set(num_cols).intersection(set(df.columns)))

scaler = StandardScaler()
df_train[num_col] = scaler.fit_transform(df_train[num_col])
df_test[num_col] = scaler.transform(df_test[num_col])

# %%
# %% [markdown]
'''
<h3>Lets validate the distribution of target column in train and test</h3>
'''
# %%
plt.figure(figsize=(16, 6))
plt.subplot(121)
sns.distplot(df_train['SalePrice'])
plt.subplot(122)
sns.distplot(df_test['SalePrice'])
plt.show()

# %% [markdown]
'''
<h3>Extract Target column</h3>
'''
# %%
y_train = df_train.pop('SalePrice')
X_train = df_train
y_test = df_test.pop('SalePrice')
X_test = df_test

# %% [markdown]
'''
<h3>I am going to use RFECV for automatic  feature elemination</h3>
'''
# %%


def feature_selector(feature_selector_obj, train_data: pd.DataFrame, train_target: pd.DataFrame):
    selector = feature_selector_obj.fit(train_data, train_target)
    display(selector.support_)
    print(f"Total selected features: {selector.support_.sum()}")
    print(
        f"Selected feature Names: {list(train_data.columns[selector.support_])}")
    display(selector.ranking_)
    return selector


# %%
# get RFECV selector
rfe_selector = feature_selector(RFECV(LinearRegression(), step=1, cv=5), X_train, y_train)
# %%
# %%
# get only usefull columns from train and test
rfe_sel_columns = list(X_train.columns[rfe_selector.support_])
X_train = X_train[rfe_sel_columns]
X_test = X_test[rfe_sel_columns]

# %% [markdown]
'''
<h3>Lets try lasso</h3>
'''
# %%
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

y_train_pred = lasso.predict(X_train)
print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lasso.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_test_pred))
# %%

# print Lasso coefficients
model_parameter = list(lasso.coef_)
model_parameter.insert(0, lasso.intercept_)
model_parameter = [round(x, 3) for x in model_parameter]
col = df_train.columns
col.insert(0, 'Constant')
list(zip(col, model_parameter))

# %% [markdown]
'''
<h3>Lets used Grid Search CV for Lasso to get best model</h3>
'''
# %%
#  Grid search CV for Lasso
lasso = Lasso()
params = {'alpha': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]}
folds = KFold(n_splits=10, shuffle=True, random_state=42)
cv_model = GridSearchCV(estimator=lasso, param_grid=params, scoring='r2',
                        cv=folds, return_train_score=True, verbose=1)
cv_model.fit(X_train, y_train)

# %%
cv_result = pd.DataFrame(cv_model.cv_results_)
cv_result['param_alpha'] = cv_result['param_alpha'].astype('float32')
cv_result
# %%
#  plotting train and test errors
plt.figure(figsize=(16, 8))
plt.plot(cv_result['param_alpha'], cv_result['mean_train_score'])
plt.plot(cv_result['param_alpha'], cv_result['mean_test_score'])
plt.xscale('log')
plt.ylabel('R2 Score')
plt.xlabel('Alpha')
plt.show()

# %%
# best alpha
cv_model.best_params_

# %%
#  train model with best alpha

lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %%
#  find lasso params
model_param = list(lasso.coef_)
model_param.insert(0, lasso.intercept_)
cols = df_train.columns
cols.insert(0, 'const')
lasso_coef = pd.DataFrame(list(zip(cols, model_param)))
lasso_coef.columns = ['Feature Name', 'Coefficient Value']
# %%
# %% [markdown]
'''
<h2>Best (TOP 15) Lasso params</h2>
'''
# %%
lasso_coef.sort_values(by='Coefficient Value', ascending=False).head(15).style.hide_index()

# %% [markdown]
'''
<h2>Ridge Regression</h2>
'''
# %%
ridge = Ridge(alpha=0.001)
ridge.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)
print(r2_score(y_train, y_train_pred))
y_test_pred = ridge.predict(X_test)
print(r2_score(y_test, y_test_pred))

# %% [markdown]
'''
Good R-square values
'''
# %%
#  Grid search CV for Ridge
ridge = Ridge()
params = {'alpha': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]}
folds = KFold(n_splits=10, shuffle=True, random_state=42)
cv_model = GridSearchCV(estimator=ridge, param_grid=params, scoring='r2',
                        cv=folds, return_train_score=True, verbose=1)
cv_model.fit(X_train, y_train)

# %%
r_cv_result = pd.DataFrame(cv_model.cv_results_)
r_cv_result['param_alpha'] = r_cv_result['param_alpha'].astype('float32')
r_cv_result.head()
# %%
#  plotting train and test errors
plt.figure(figsize=(16, 8))
plt.plot(r_cv_result['param_alpha'], r_cv_result['mean_train_score'])
plt.plot(r_cv_result['param_alpha'], r_cv_result['mean_test_score'])
plt.xlabel('Alpha')
# plt.xscale('log')
plt.ylabel('R2 Score')
plt.show()

# %%
# best alpha
cv_model.best_params_

# %%
#  train model with best alpha

ridge = Ridge(alpha=20.0)
ridge.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %%
#  find ridge params
model_param = list(ridge.coef_)
model_param.insert(0, ridge.intercept_)
cols = df_train.columns
cols.insert(0, 'const')
ridge_coef = pd.DataFrame(list(zip(cols, model_param)))
ridge_coef.columns = ['Feature Name', 'Coefficient Value']
# %%
# %% [markdown]
'''
<h2>Best (TOP 15) Ridge params</h2>
'''
# %%
ridge_coef.sort_values(by='Coefficient Value', ascending=False).head(15).style.hide_index()

# %% [markdown]
'''
<h4>We can see that R-Squared are almost same (little better in ridge 0.903 > 0.899) but Lasso has advantage of being simpler model. So here I am going to choose <b>Lasso</b> over Ridge</h4>
'''

# %% [markdown]
'''
<h4>Model for this analysis</h4>
'''
# %%
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %% [markdown]
'''
<h2>With doubling of alpha</h2>
'''

# %% [markdown]
'''
With Lasso
'''

# %%
#  train model with best alpha

lasso = Lasso(alpha=0.002)
lasso.fit(X_train, y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %%
#  find lasso params
model_param = list(lasso.coef_)
model_param.insert(0, lasso.intercept_)
cols = df_train.columns
cols.insert(0, 'const')
lasso_coef = pd.DataFrame(list(zip(cols, model_param)))
lasso_coef.columns = ['Feature Name', 'Coefficient Value']
# %%
lasso_coef.sort_values(by='Coefficient Value', ascending=False).head(15).style.hide_index()
# %% [markdown]
'''
With Ridge 
'''

# %%

ridge = Ridge(alpha=40.0)
ridge.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %%
#  find ridge params
model_param = list(ridge.coef_)
model_param.insert(0, ridge.intercept_)
cols = df_train.columns
cols.insert(0, 'const')
ridge_coef = pd.DataFrame(list(zip(cols, model_param)))
ridge_coef.columns = ['Feature Name', 'Coefficient Value']

# %%
ridge_coef.sort_values(by='Coefficient Value', ascending=False).head(15).style.hide_index()

# %% [markdown]
'''
Remove Top 5 features and then get top parametesr
'''

# %% [markdown]
'''
For Lasso
'''
# %%
top_lasso_featutes = ["YrSold_Old", "LowQualFinSF", "GarageCond_Others", "GarageYrBlt_Old", "Exterior1st_Others"]

X_train_dropped = X_train.drop(top_lasso_featutes, axis=1)
X_test_dropped = X_test.drop(top_lasso_featutes, axis=1)
# %%
#  train model with best alpha

lasso = Lasso(alpha=0.001)
lasso.fit(X_train_dropped, y_train)

y_train_pred = lasso.predict(X_train_dropped)
y_test_pred = lasso.predict(X_test_dropped)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %%
#  find lasso params
model_param = list(lasso.coef_)
model_param.insert(0, lasso.intercept_)
cols = X_train_dropped.columns
cols.insert(0, 'const')
lasso_coef = pd.DataFrame(list(zip(cols, model_param)))
lasso_coef.columns = ['Feature Name', 'Coefficient Value']
# %%
lasso_coef.sort_values(by='Coefficient Value', ascending=False).head(15).style.hide_index()
# %% [markdown]
'''
For Ridge
'''
# %%
top_ridge_featutes = ["YrSold_Old", "BsmtUnfSF", "LowQualFinSF", "GarageQual_Others", "Neighborhood_NridgHt"]
X_train_dropped = X_train.drop(top_ridge_featutes, axis=1)
X_test_dropped = X_test.drop(top_ridge_featutes, axis=1)
# %%

ridge = Ridge(alpha=20.0)
ridge.fit(X_train_dropped, y_train)

y_train_pred = ridge.predict(X_train_dropped)
y_test_pred = ridge.predict(X_test_dropped)

print(r2_score(y_true=y_train, y_pred=y_train_pred))
print(r2_score(y_true=y_test, y_pred=y_test_pred))

# %%
#  find ridge params
model_param = list(ridge.coef_)
model_param.insert(0, ridge.intercept_)
cols = X_train_dropped.columns
cols.insert(0, 'const')
ridge_coef = pd.DataFrame(list(zip(cols, model_param)))
ridge_coef.columns = ['Feature Name', 'Coefficient Value']

# %%
ridge_coef.sort_values(by='Coefficient Value', ascending=False).head(15).style.hide_index()

# %%
