# %% [markdown]
'''
# Imports
'''

# %%
from sklearn.feature_selection import RFECV, RFE
from functools import reduce
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from imblearn.metrics import sensitivity_specificity_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from pandas._config.config import option_context
from pandas_profiling import ProfileReport
import numpy as np
import pandas as pd
import warnings
from IPython.display import display

import plotly.figure_factory as ff
import plotly.graph_objects as go
from enum import Enum, auto
from typing import List, Sequence, Tuple
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from imblearn.over_sampling import SMOTE
# %%
warnings.filterwarnings('ignore')
# %% [markdown]
'''
<h3>Pandas settings</h3>
'''
# %%
pd.options.display.max_columns = None
pd.options.display.max_rows = 500
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3


# %% [markdown]
'''
 # Utility Functions
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
        showlegend: bool = False,
        log_x=False,
        log_y=False,
        * args,
        **kwargs) -> None:
    """Bar plots a interger series

    Args:
        series (pd.Series): series to be plotted
        title (str): graph title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        graph_type (GraphType, optional): graph type
        showlegend (bool, optional): default False
        log_x (bool, optional): default False
        log_y (bool, optional): default False
    """
    labels = {"x": xlabel, "y": ylabel}
    fig = None
    if graph_type is None or graph_type == GraphType.BAR:
        fig = px.bar(x=series.index, y=series, color=series.index,
                     title=title, labels=labels, log_x=log_x, log_y=log_y, *args, **kwargs)

    if graph_type == GraphType.LINE:
        px.scatter(x=series.index, y=series, title=title, labels=labels, color=series.index, *args,
                   **kwargs)

    fig.update_layout(showlegend=showlegend)
    fig.show()


def get_univariate_cat_plot_strs(value: str) -> Tuple[str, str, str]:
    """Creates graph title, x-axis text and y-axis text for given value

    Args:
        value (str): column name

    Returns:
        Tuple[str, str, str]: title, x-axis text and y-axis text
    """
    full_name = " ".join([data_dict_as_dict.get(v.upper(), v.title()) for v in value.split("_")])
    if len(full_name) > 30:
        full_name = value
    count_str = full_name + ' Count'
    return count_str + ' Plot', full_name, count_str


def plot_cat_data(c: str, value_counts_ser: pd.Series, *args, **kwargs):
    """Plots the value count series

    Args:
        c ([str]): column name
        value_counts_ser ([pd.Series]): value counts series
    """
    t, xl, yl = get_univariate_cat_plot_strs(c)
    plot_univariate_series(value_counts_ser, t, xl, yl, *args, **kwargs)


def plot_univariate_categorical_columns(categorical_cols: Sequence[str], dataframe: pd.DataFrame, plot_limit: int = 30, *args, **kwargs) -> None:
    """plots categorical variable bars

    Args:
        categorical_cols (Sequence[str]): categorical columns
        dataframe (pd.DataFrame): DataFrame
    """
    for c in categorical_cols:
        value_counts_ser = dataframe[c].value_counts()
        cnt_len = len(value_counts_ser)
        if cnt_len > 1 and cnt_len < plot_limit:
            plot_cat_data(c, value_counts_ser, *args, **kwargs)


def plot_value_counts_frame(dataframe: pd.DataFrame, plot_limit: int = 30, *args, **kwargs) -> None:
    """plots categorical variable bars

    Args:
        dataframe (pd.DataFrame): dataframe with Column and Value Counts
        plot_limit (int, optional): Plot will be generated of catgory count is less than, Defaults to 30.
    """

    for i in range(len(dataframe)):
        current_row = dataframe.iloc[i]
        c = current_row["Column"]
        value_counts_ser = current_row['ValueCounts']
        cnt_len = len(value_counts_ser)
        if cnt_len > 1 and cnt_len <= plot_limit:
            plot_cat_data(c, value_counts_ser, *args, **kwargs)


# %%
#  Read the data and Data Dictionary
churn_data = pd.read_csv("telecom_churn_data.csv")
data_dict = pd.read_excel("Data+Dictionary-+Telecom+Churn+Case+Study.xlsx")
data_dict = data_dict.sort_values(by='Acronyms    ')
# %%
data_dict_styled = data_dict.style.set_properties(**{'text-align': 'left'})
data_dict_styled.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
data_dict_styled.hide_index()
# %% [markdown]
'''
<h2>Data overview</h2>
'''
# %%
churn_data.shape
# %%
churn_data.sample(20)

# %%
", ".join(churn_data.columns)

# %%
churn_data.info(verbose=1)

# %%
churn_data.describe(include='all').T
# %%
#  Null percents


def print_null_percents(frame: pd.DataFrame, full: bool = False, display_cols=True):
    """Prints null columns perdent and count

    Args:
        frame (pd.DataFrame):Dataframe where null needs to be counted
        full (bool, optional): show all columns. Defaults to False.
        display_cols (bool, optional): show columns or not. Defaults to True.
    """
    null_counts = frame.isna().sum()
    if not full:
        null_counts = null_counts[null_counts > 0]
    if display_cols:
        display(round((null_counts/frame.shape[0])*100, 2).sort_values(ascending=False))
    print(f"Columns count with null: {len(null_counts[null_counts > 0])}")


# %%
print_null_percents(churn_data, True)
# %% [markdown]
'''
# EDA
'''
# %%
# duplicate Rows Counts

len(churn_data) - len(churn_data.drop_duplicates())

# %%
#  value counts DataFrame
vc = []
for c in churn_data.columns:
    vc.append((c, churn_data[c].value_counts()))
vc = pd.DataFrame(vc, columns=['Column', "ValueCounts"])


# %% [markdown]
'''
# Data cleaning
'''
# %%
# Make column names consistent
churn_data.rename({'aug_vbc_3g': 'vbc_3g_8', 'jul_vbc_3g': 'vbc_3g_7', 'jun_vbc_3g': 'vbc_3g_6', 'sep_vbc_3g': 'vbc_3g_9'}, axis=1, inplace=True)
# %%
# pr = ProfileReport(df, minimal=True)
# pr.to_widgets()

# %%

date_cols = ['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9', 'date_of_last_rech_6', 'date_of_last_rech_7',
             'date_of_last_rech_8', 'date_of_last_rech_9', 'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8', 'date_of_last_rech_data_9']
cat_cols = ['fb_user_6', 'fb_user_7', 'fb_user_8', 'fb_user_9', 'night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8', 'night_pck_user_9']
identity_cols = ['mobile_number', 'circle_id']
num_cols = [column for column in churn_data.columns if column not in identity_cols + date_cols + cat_cols]

# %%
# print the number of columns in each list
print(f"Number of ID cols: {len(identity_cols)}\nNumber of Date cols:{len(date_cols)}\nNumber of Numeric cols:{len(num_cols)}\nNumber of Category cols:{len(cat_cols)}")
# %%
# convert date time data type
for dc in date_cols:
    churn_data[dc] = pd.to_datetime(churn_data[dc])

# Convert Categorical columns -- Need to set after imputation (as fill na for categorical values only allow values one from categorical)
# churn_data[cat_cols] = churn_data[cat_cols].astype('category')

# %%
#  Max and min dates in data
print(f"Max Date:\n{churn_data[date_cols].max().max()}")
print("-"*25)
print(f"Min Date:\n{churn_data[date_cols].min().min()}")

# %%
#  Max and min dates in data
print(f"Max Dates:\n{churn_data[date_cols].max()}")
print("-"*25)
print(f"Min Dates:\n{churn_data[date_cols].min()}")

# %%
# Print value counts


def print_value_counts(frame: pd.DataFrame) -> None:
    frame = frame.sort_values(by='Column')
    for i in range(len(frame)):
        print("="*25)
        print(frame.iloc[i]['ValueCounts'])


# %%
# Cols with single value
single_valued_cols = vc[vc.apply(lambda r: len(r["ValueCounts"]) == 1, axis=1)]
print_value_counts(single_valued_cols)
# %%

print(single_valued_cols['Column'])
print(len(single_valued_cols))
# %%
# Unique value counts less than 30(No specific reason to choose 30, after 30 categories it make sense to bucketization)
cols_less_30_values = vc[vc.apply(lambda r: len(r["ValueCounts"]) > 1 and len(r["ValueCounts"]) <= 30, axis=1)]
print_value_counts(cols_less_30_values)


# %%
data_dict_as_dict = {}
for i in range(len(data_dict)):
    data_dict_as_dict[data_dict.iloc[i]['Acronyms    '].strip()] = data_dict.iloc[i]['Descriptions']

data_dict_as_dict['6'] = ' for Jun'
data_dict_as_dict['7'] = ' for Jul'
data_dict_as_dict['8'] = ' for Aug'
data_dict_as_dict['9'] = ' for Sep'

# %%
# plot_value_counts_frame(cols_less_30_values.sort_values(by='Column'), log_y=True)

# %% [markdown]
'''
# Imputation
'''

# %% [markdown]
'''
# Categorical value imputation
'''
# %%
# print categorical value counts
for cc in cat_cols:
    print(churn_data[cc].value_counts())

# %%
# Check for nulls in cate columns
churn_data[cat_cols].isna().sum()

# %%
#  Fill NAs and convert to categorical
churn_data[cat_cols] = churn_data[cat_cols].apply(lambda x: x.fillna(-1))
churn_data[cat_cols] = churn_data[cat_cols].astype(int)
churn_data[cat_cols] = churn_data[cat_cols].astype('category')


# %%
print("Missing value ratio in categorical columns:\n")
display(churn_data[cat_cols].isnull().sum()*100/churn_data.shape[1])

# %%
# data preview
churn_data[cat_cols]

# %% [markdown]
'''
# Imputing Numerics
'''
# %%
# Considering if recharge value is blank then person did not recharge and we can put zero for recharge as this is currency and defauld zero is good option
zero_impute = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
               'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
               'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9']
# %%
# Fill recharge amount NAs to zero
churn_data[zero_impute] = churn_data[zero_impute].apply(lambda x: x.fillna(0))

# %%
print("Missing value ratio in zero imputed columns:\n")
display(churn_data[zero_impute].isnull().sum()*100/churn_data.shape[1])

# %%
# Re-look on missing percentage
print_null_percents(churn_data)

# %%
# columns with more than 70% missing values
MISSING_PERCENT_THRESHOLD = 70
null_counts = churn_data.isna().sum()
null_counts = null_counts[null_counts > 0]
null_percentage = round((null_counts/churn_data.shape[0])*100, 2).sort_values(ascending=False)
null_percentage = null_percentage[null_percentage >= MISSING_PERCENT_THRESHOLD]
display(null_percentage)

# %%
#  lets have a look on missing data columns
churn_data[null_percentage.index].sample(20)

# %%
missing_above_threshold = churn_data[null_percentage.index].describe(include='all').T
missing_above_threshold["missing_per"] = null_percentage
missing_above_threshold
# %% [markdown]
'''
# Dropping columns
'''
# %%
# Dropping Id columns
print(churn_data.shape)
churn_data.drop(identity_cols, axis=1, inplace=True)
churn_data.shape
# %%
# Dropping columns wit high missing values
print(churn_data.shape)
churn_data.drop(null_percentage.index, axis=1, inplace=True)

churn_data.shape
# %%
# One valued columns as they will not produce any value
one_valued_columns = [c for c in churn_data.columns if len(churn_data[c].value_counts()) == 1]
one_valued_columns

# %%
#  Dropping single valued columns
print(churn_data.shape)
churn_data.drop(one_valued_columns, axis=1, inplace=True)
churn_data.shape
# %%
#  Dropping date columns columns
print(churn_data.shape)
churn_data.drop(['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9'], axis=1, inplace=True)
churn_data.shape

# %%
churn_data.info(verbose=1)

# %% [markdown]
'''
# Imputing with IterativeImputer imputer
'''

# %%
non_impute_cols = ['night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8', 'night_pck_user_9', 'fb_user_6', 'fb_user_7', 'fb_user_8', 'fb_user_9']
# %%
impute_ds = churn_data.drop(non_impute_cols, axis=1)

iterative_imputer = IterativeImputer(random_state=997, max_iter=1)
churn_data_imputed = iterative_imputer.fit_transform(impute_ds)
churn_data_imputed = pd.DataFrame(churn_data_imputed, columns=impute_ds.columns)
print_null_percents(churn_data_imputed, full=True)


# %%
# merge non imputed columns to imputed columns
churn_data_imputed = pd.concat([churn_data_imputed, churn_data[non_impute_cols]], axis=1)
churn_data_imputed.shape

# %%
churn_data_imputed.isnull().sum().sum()
# %% [markdown]
'''
# Filtering
'''

# %% [markdown]
'''
Those who have recharged with an amount more than or equal to X, where X is the **70th percentile ** of the **average recharge amount** in the **first two months** (the good phase).
'''
# %%
# Calculate data recharge amount
churn_data_imputed['total_data_rech_6'] = churn_data_imputed['total_rech_data_6'] * churn_data_imputed['av_rech_amt_data_6']
churn_data_imputed['total_data_rech_7'] = churn_data_imputed['total_rech_data_7'] * churn_data_imputed['av_rech_amt_data_7']

# %%
# calculate total recharge amount
churn_data_imputed['amt_data_6'] = churn_data_imputed['total_rech_amt_6'] + churn_data_imputed['total_data_rech_6']
churn_data_imputed['amt_data_7'] = churn_data_imputed['total_rech_amt_7'] + churn_data_imputed['total_data_rech_7']

# %%
# calculate mean of month 6 and 7
churn_data_imputed['av_amt_data_6_7'] = (churn_data_imputed['amt_data_6'] + churn_data_imputed['amt_data_7'])/2

# %%
# threshold value
threshold_value = churn_data_imputed['av_amt_data_6_7'].quantile(0.7)
print(f"Threshold value to filter the users is: {threshold_value}")

# %%
high_value_users = churn_data_imputed[churn_data_imputed['av_amt_data_6_7'] >= threshold_value]
high_value_users = high_value_users.reset_index()
high_value_users.shape  # (~29.9k rows)

# %%
churn_data_imputed.isnull().sum().sum()
# %% [markdown]
'''
Calculate churn based on total_ic_mou_9,total_og_mou_9,vol_2g_mb_9,vol_3g_mb_9
'''
# %%
# Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase.
high_value_users['churn'] = high_value_users.apply(lambda x:  1 if x['total_ic_mou_9'] + x['total_og_mou_9']+x['vol_2g_mb_9']+x['vol_3g_mb_9'] == 0 else 0, axis=1)
high_value_users['churn'] = high_value_users['churn'].astype('category')
high_value_users['churn'].value_counts()

# %% [markdown]
'''
# Remove all the attributes corresponding to the churn phase
'''
# %%
# get churn month columns
churn_phase_cols = [c for c in high_value_users.columns if str(c).endswith("_9")]
churn_phase_cols
# %%
# drop churn month columns
high_value_users = high_value_users.drop(churn_phase_cols, axis=1)
high_value_users.shape

# %%
high_value_users.info(verbose=True)

# %% [markdown]
'''#### Plotting utils'''
# %%
# Plotting functions


def plot_dist(data_frame: pd.DataFrame, cols_to_plot: List[str], merge_all: bool = False) -> None:
    if merge_all:
        fig = ff.create_distplot(data_frame, cols_to_plot)
        fig.show()
    else:
        for i, c in enumerate(cols_to_plot):
            fig = ff.create_distplot([data_frame[i]], [c])
            fig.update_layout(title_text=f"Dist plot for {c}")
            fig.show()


def plot_box(df: pd.DataFrame, x: str, y: str) -> None:
    fig = px.box(df, x=x, y=y, color=x)
    fig.show()


def getdtype(col):
    if col.dtype == np.int64 or col.dtype == np.float64:
        return 'num'
    elif col.dtype == 'category':
        return 'cat'


def plot_two_variables(df, x, y):
    if getdtype(df[x]) == 'num' and getdtype(df[y]) == 'num':
        fig = px.scatter(df, x=x, y=y, trendline="ols")
        fig.show()
    elif (getdtype(df[x]) == 'cat' and getdtype(df[y]) == 'num'):
        plot_box(df, x, y)
    elif (getdtype(df[x]) == 'num' and getdtype(df[y]) == 'cat'):
        plot_box(df, y, x)


# %%
cat_col_names = high_value_users.select_dtypes(include='category').columns
cat_col_names
# %%
# plot_univariate_categorical_columns(cat_col_names, high_value_users, showlegend=True)
# %%
base_num_cols = ['arpu_', 'onnet_mou_', 'offnet_mou_', 'roam_ic_mou_', 'roam_og_mou_', 'loc_og_t2t_mou_', 'loc_og_t2m_mou_', 'loc_og_t2f_mou_', 'loc_og_t2c_mou_', 'loc_og_mou_', 'std_og_t2t_mou_', 'std_og_t2m_mou_', 'std_og_t2f_mou_', 'std_og_mou_', 'isd_og_mou_', 'spl_og_mou_', 'og_others_', 'total_og_mou_', 'loc_ic_t2t_mou_', 'loc_ic_t2m_mou_', 'loc_ic_t2f_mou_', 'loc_ic_mou_',
                 'std_ic_t2t_mou_', 'std_ic_t2m_mou_', 'std_ic_t2f_mou_', 'std_ic_mou_', 'total_ic_mou_', 'spl_ic_mou_', 'isd_ic_mou_', 'ic_others_', 'total_rech_num_', 'total_rech_amt_', 'max_rech_amt_', 'last_day_rch_amt_', 'total_rech_data_', 'max_rech_data_', 'av_rech_amt_data_', 'vol_2g_mb_', 'vol_3g_mb_', 'monthly_2g_', 'sachet_2g_', 'monthly_3g_', 'sachet_3g_', 'vbc_3g_']

# %%
for base_num_col in base_num_cols[:1]:
    cols = [base_num_col+m for m in ["6", '7', '8']]
    data = [high_value_users[c] for c in cols]
    plot_dist(data, cols, True)

# %%
# TODO: puu some more columns
plot_two_variables(high_value_users, "churn", "aon")

# %% [markdown]
'''
# Outlier treatment
'''

# %%


def treat_outlier(ser: pd.Series) -> pd.Series:
    mean_value = ser.mean()
    std = ser.std()
    lower, upper = mean_value - 3 * std,  mean_value + 3 * std
    ser[ser < lower] = lower
    ser[ser > upper] = upper
    return ser


# %%
hvc_num_cols = high_value_users.select_dtypes(include=np.number).columns

# %%
# Treat outliers of numeric columns

high_value_users[hvc_num_cols] = high_value_users[hvc_num_cols].apply(treat_outlier, axis=0)

# %%
# Convert churn column to numeric for modeling
high_value_users['churn'] = pd.to_numeric(high_value_users['churn'])

# %%
# Churn to Non churn percentage
(high_value_users['churn'].value_counts()/high_value_users.shape[0])*100

# %%
high_value_users.isnull().sum().sum()
# %% [markdown]
'''
## Feature extraction
'''

# %%

# Feature extraction functions


def get_ratio(x, y):
    if x == y:
        return 1
    elif y == 0:
        return x
    else:
        return x/y


def is_increasing(df: pd.DataFrame, base_col: str) -> pd.Series:
    cols = [base_col+m for m in ["6", '7', '8']]
    return df.apply(lambda row: 1 if (row[cols[0]] <= row[cols[1]]) and (row[cols[1]] <= row[cols[2]]) else 0, axis=1)


def is_decreasing(df: pd.DataFrame, base_col: str) -> pd.Series:
    cols = [base_col+m for m in ["6", '7', '8']]
    return df.apply(lambda row: 1 if (row[cols[0]] >= row[cols[1]]) and (row[cols[1]] >= row[cols[2]]) else 0, axis=1)


# %%
# shape of frame before feature extraction
high_value_users.shape

# %%
# Feature extraction
for base_num_col in base_num_cols:
    cols = [base_num_col+m for m in ["6", '7', '8']]
    val_6 = high_value_users[cols[0]]
    val_7 = high_value_users[cols[1]]
    val_8 = high_value_users[cols[2]]

    # Ratios
    high_value_users[base_num_col+"ratio_6_7"] = high_value_users.apply(lambda row: get_ratio(row[cols[0]], row[cols[1]]), axis=1)
    high_value_users[base_num_col+"ratio_7_8"] = high_value_users.apply(lambda row: get_ratio(row[cols[1]], row[cols[2]]), axis=1)

    #  Stats
    high_value_users[base_num_col+"sum"] = val_6+val_7+val_8
    high_value_users[base_num_col+"mean"] = (val_6+val_7+val_8)/3
    high_value_users[base_num_col+"std"] = high_value_users[cols].std(axis=1)

    # is_increasing
    high_value_users[base_num_col+"is_increasing"] = is_increasing(high_value_users, base_num_col)
    # is_decreasing
    high_value_users[base_num_col+"is_decreasing"] = is_decreasing(high_value_users, base_num_col)
# %%
# shape of frame before feature extraction
high_value_users.shape
# %%
# Is incoming more than outgoing
incoming_cols = [c for c in high_value_users.columns if "_ic_" in c]
outgoing_cols = [c for c in high_value_users.columns if "_og_" in c]
high_value_users['is_high_incoming'] = high_value_users[incoming_cols].apply(lambda row: sum(row), axis=1) > high_value_users[outgoing_cols].apply(lambda row: sum(row), axis=1)

high_value_users['is_high_incoming'] = high_value_users['is_high_incoming'].map({True: 1, False: 0})
# %%
# shape of frame before feature extraction
high_value_users.shape
# %%
high_value_users.isnull().sum().sum()

# %%
# Convert categorical columns to
for c in high_value_users.select_dtypes('category').columns:
    high_value_users[c] = high_value_users[c].astype(np.float64)

# %%
high_value_users.isnull().sum().sum()
# %%
high_value_users.info(verbose=True)
# %% [markdown]
'''
## Modeling
'''

# %%
X = high_value_users.drop("churn", axis=1)
y = high_value_users['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=997, stratify=y)

# %%
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %%
# Churn to Non churn percentage in train and test
print((y_train.value_counts()/len(y_train))*100)
print(y_test.value_counts()/len(y_test)*100)
# %%
X_train.isnull().sum().sum()
# %% [markdown]
'''
### Scale/Standardization/Normalization
'''
# %%
# Scale numeric columns
num_cols = X_train.select_dtypes(np.number).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
# %%
# Scale Test data
X_test[num_cols] = scaler.transform(X_test[num_cols])
# %%
X_train.isnull().sum().sum()
# %% [markdown]
'''
### Class balancing
'''
# %%
# Smoting
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
# %%
X_train_bal.isnull().sum().sum()
# %%
# Doing PCA
pca = PCA()
pca.fit_transform(X_train_bal)

# %%
# Explained variance of PCA components
print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))

# %%

# plot feature variance
features = range(pca.n_components_)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
plt.figure(figsize=(10, 10))
plt.plot(cumulative_variance)
plt.show()
# %%
VARIANCE_FRACTION = 0.9
# %%

# PCA with 90% variance
pca = PCA(VARIANCE_FRACTION)
pca.fit_transform(X_train_bal)
print(f"{VARIANCE_FRACTION*100}% variance is explained by {pca.n_components_} principle components")


# %%
all_metrics = []
# %% [markdown]
'''
## Classification with PCA

### 1. Logistic Regression
'''

# %%
pipeline = Pipeline([("pca", PCA(VARIANCE_FRACTION)),
                     ("logistic", LogisticRegression())])


# %%
# fit model
pipeline.fit(X_train_bal, y_train_bal)


# %%
# check score on train data
pipeline.score(X_train_bal, y_train_bal)

# %% [markdown]
'''
### evaluate on test data
'''
# %%
y_pred = pipeline.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print(f"Sensitivity: {round(sensitivity, 2)}")
print(f"Specificity: {round(specificity, 2)}")

# %%
# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print(f"AUC: {round(roc_auc_score(y_test, y_pred_prob), 2)}")
# %%

all_metrics.append(["Logistic Regression", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])

# %% [markdown]
'''
### Logistic Regression with hyper parameter tuning
'''
# %%

# 5 fold randomized search CV
model = GridSearchCV(
    estimator=Pipeline([('pca', PCA()), ("logistic", LogisticRegression())]),
    cv=KFold(n_splits=5, shuffle=True, random_state=997),
    param_grid={'pca__n_components': [120, 130, 140, 150, 160, 180, 200], 'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']},
    scoring='roc_auc',
    n_jobs=5,
    verbose=1)
# %%
# train the model
model.fit(X_train, y_train)
# %%
# cross validation results
pd.DataFrame(model.cv_results_)

# %%
# Best score and params
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)

# %%
# check score on train data
model.score(X_train_bal, y_train_bal)

# %% [markdown]
'''
### evaluate on test data
'''
# %%
y_pred = model.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print(f"Sensitivity: {round(sensitivity, 2)}")
print(f"Specificity: {round(specificity, 2)}")

# %%
# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC: {round(roc_auc_score(y_test, y_pred_prob), 2)}")
# %%

all_metrics.append(["Logistic Regression With Grid Search", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])


# %% [markdown]
'''
### 2. Random forest
'''

# %%
pipeline = Pipeline([("pca", PCA(VARIANCE_FRACTION)),
                     ("rf", RandomForestClassifier())])


# %%
# fit model
pipeline.fit(X_train_bal, y_train_bal)


# %%
# check score on train data
pipeline.score(X_train_bal, y_train_bal)

# %% [markdown]
'''
### evaluate on test data
'''
# %%
y_pred = pipeline.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print(f"Sensitivity: {round(sensitivity, 2)}")
print(f"Specificity: {round(specificity, 2)}")

# %%
# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print(f"AUC: {round(roc_auc_score(y_test, y_pred_prob), 2)}")
# %%

all_metrics.append(["Random Forest", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])

# %% [markdown]
'''
### Random Forest with hyper parameter tuning
'''
# %%

# 5 fold randomized search CV
model = RandomizedSearchCV(
    estimator=Pipeline([('pca', PCA()), ("rf", RandomForestClassifier())]),
    cv=KFold(n_splits=5, shuffle=True, random_state=997),
    param_distributions={'pca__n_components': [120, 130, 140, 150, 160, 180, 200],
                         'rf__n_estimators': [10, 20, 50, 100, 150, 200],
                         'rf__max_depth': [5, 10, 20, 50, 100],
                         'rf__min_samples_split': [2, 5, 10, 20, 50, 100],
                         'rf__min_samples_leaf': [2, 5, 10, 20, 50, 100]},
    scoring='roc_auc',
    n_jobs=5,
    verbose=1)
# %%
# train the model
model.fit(X_train, y_train)
# %%
# cross validation results
pd.DataFrame(model.cv_results_)

# %%
# Best score and params
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)

# %%
# check score on train data
model.score(X_train_bal, y_train_bal)

# %% [markdown]
'''
### evaluate on test data
'''
# %%
y_pred = model.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print(f"Sensitivity: {round(sensitivity, 2)}")
print(f"Specificity: {round(specificity, 2)}")

# %%
# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC: {round(roc_auc_score(y_test, y_pred_prob), 2)}")
# %%

all_metrics.append(["Random Forest With Randomized Search", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])

# %% [markdown]
'''
### 3. Gradient Boosting
'''
# %%
pipeline = Pipeline([("pca", PCA(VARIANCE_FRACTION)),
                     ("gb", GradientBoostingClassifier())])


# %%
# fit model
pipeline.fit(X_train_bal, y_train_bal)


# %%
# check score on train data
pipeline.score(X_train_bal, y_train_bal)

# %% [markdown]
'''
### evaluate on test data
'''
# %%
y_pred = pipeline.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print(f"Sensitivity: {round(sensitivity, 2)}")
print(f"Specificity: {round(specificity, 2)}")

# %%
# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print(f"AUC: {round(roc_auc_score(y_test, y_pred_prob), 2)}")
# %%

all_metrics.append(["Gradient Boosting", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])

# %% [markdown]
'''
### Gradient Boosting with hyper parameter tuning
'''
# %%

# 5 fold randomized search CV
model = RandomizedSearchCV(
    estimator=Pipeline([('pca', PCA()), ("gb", GradientBoostingClassifier())]),
    cv=KFold(n_splits=5, shuffle=True, random_state=997),
    param_distributions={'pca__n_components': [120, 130, 140, 150, 160, 180, 200],
                         'gb__loss': ['deviance', 'exponential'],
                         'gb__learning_rate': [0.01, 0.1, 1, 10],
                         'gb__n_estimators': [10, 20, 50, 100, 150, 200]},
    scoring='roc_auc',
    n_jobs=5,
    verbose=1)
# %%
# train the model
model.fit(X_train, y_train)
# %%
# cross validation results
pd.DataFrame(model.cv_results_)

# %%
# Best score and params
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)

# %%
# check score on train data
model.score(X_train_bal, y_train_bal)

# %% [markdown]
'''
### evaluate on test data
'''
# %%
y_pred = model.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

# %%

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print(f"Sensitivity: {round(sensitivity, 2)}")
print(f"Specificity: {round(specificity, 2)}")

# %%
# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC: {round(roc_auc_score(y_test, y_pred_prob), 2)}")
# %%

all_metrics.append(["Gradient Boosting With Randomized Search", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])


# %%
# Performance comparisions
pef_comparision_matrix = pd.DataFrame(all_metrics, columns=["Algo", "Sensitivity", "Specificity", "AUC"])
pef_comparision_matrix.style.hide_index()
# %% [markdown]
'''
Model for Feature Importance
'''
# %%
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=20, step=1)
selector = selector.fit(X_train_bal, y_train_bal)
display(selector.support_)
display(selector.ranking_)
# %%
features = pd.DataFrame()
# %%
features["Column"] = X_train_bal.columns
features["Rank"] = selector.ranking_

# %%
# Set importance and Coefficient
features_importance = features[selector.support_]
features_importance.reset_index(drop=True, inplace=True)
importance_frame = features_importance.reindex()
importance_frame["Coefficient"] = list(selector.estimator_.coef_[0])
importance_frame["Importance"] = list(np.abs(selector.estimator_.coef_)[0])
importance_frame = importance_frame.sort_values(by="Importance", ascending=False)
importance_frame["Direction"] = importance_frame["Coefficient"].apply(lambda x: "Pos" if x > 0 else "Neg")
# %%
importance_frame[["Column", "Importance", "Direction"]].style.hide_index()
# %%
