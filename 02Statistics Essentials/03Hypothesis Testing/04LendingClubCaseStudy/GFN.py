# %% [markdown]
'''
# LendingClub data analysis

<h2>Imports</h2>
'''
# %%
from enum import Enum, auto
from typing import Callable, Tuple, TypeVar
from matplotlib.pyplot import axis, show, title, xlabel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# %%
sns.__version__

# %% [markdown]
'''
**Must be 0.11.0**
'''
# %% [markdown]
'''
<h2>Data load</h2>
'''
# %%

loanDf = pd.read_csv("loan.csv", encoding='ISO-8859-1')
# %% [markdown]
'''
<h3>Global variables/options</h3>
'''
pd.options.display.max_columns = 100
pd.options.display.max_rows = 5000

# %% [markdown]
'''
<h3>Data understanding</h3>
'''
# %%
loanDf.shape
# %%
loanDf.describe()
# %%
loanDf.head()
# %%
loanDf.info()

# %% [markdown]
'''
<h4> We can divide columns into 3 categories</h4>
- Applicant information
- Loan Information
- User behavior
'''
# %%
loanDf.dtypes

# %% [markdown]
'''
**Except just by seeing that many rows have above 50 columns null, we don't get any information**
'''
# %%
loanDf.isnull().sum(axis=1).describe()

# %% [markdown]
'''
<h3>Target Column : loan_status</h3>
'''
# %% [markdown]
'''
<h3>We'll drop all Current Running loans as they caanot be used for analysis, because we cannot make decision(currently) if they are going to be fully paid or defaulted</h3>
'''
# %%
display(loanDf.shape)
loanDf = loanDf[loanDf['loan_status'] != 'Current']
display(loanDf.shape)
# %% [markdown]
'''
<h3>Prepare target Column</h3>
'''
# %%
loanDf['loan_status'] = loanDf['loan_status'].apply(
    lambda ls: 1 if ls == 'Charged Off' else 0)


# %% [markdown]
'''
<h2>Data Cleaning</h2>
'''
# %%
loanDf.isnull().any()
# %%
loanDf.isnull().sum()
# %%
loanDf.isnull().sum(axis=1)

# %% [markdown]
'''
<h4>Get the percentage of missing values in each column</h4>
'''

# %%
null_percent = round(100*(loanDf.isnull().sum()/len(loanDf.index)), 2)
null_percent
# %% [markdown]
'''
**Columns with more than 60% nulls**
'''
# %%
null_percent[null_percent > 60]

# %% [markdown]
'''
**We'll drop these columns as they are part user behavior and almost all nulls**
'''
# %%
columns_toRemove = null_percent.loc[null_percent > 60].index
columns_toRemove

# %% [markdown]
# #### Let us remove the columns having all missing values greater than 70 Percent

# %%
filtered_df = loanDf.drop(columns_toRemove, axis=1)


# %%
filtered_df.shape

# %% [markdown]
# ##### Let us now deal with other missing values within the remaining data

# %%
moreMissingCols = round(
    100*(filtered_df.isnull().sum()/len(filtered_df.index)), 2).sort_values(ascending=False)
moreMissingCols[moreMissingCols > 0]


# %% [markdown]
'''
**We'll drop below mentioned columns because either they are part of user behavior or very less informative for EDA**
'''
# %%
filtered_df = filtered_df.drop(['desc', 'collections_12_mths_ex_med', 'total_rec_int', 'total_pymnt',
                                'chargeoff_within_12_mths', 'tax_liens', 'last_credit_pull_d', 'pub_rec_bankruptcies', 'recoveries', 'collection_recovery_fee', 'total_rec_late_fee', 'total_rec_prncp', 'last_pymnt_amnt', 'total_pymnt', 'delinq_2yrs'], axis=1)


# %%
moreMissingCols = round(
    100*(filtered_df.isnull().sum()/len(filtered_df.index)), 2).sort_values(ascending=False)
moreMissingCols[moreMissingCols > 0]

# %% [markdown]
'''
<h3>Columns which could be imputed</h3>
| Column Name | Imputation Technique | Reason |
|-|-|-|
| title | mode |Categorical |
| emp_title | mode |Categorical |
| emp_length | mode |We are considering as categorical, could be mean as there are no outliers |
| last_pymnt_d | TBD |Depends on business decision |
| revol_util | mean  or median|  based on data data distribution or business decision  |

<h4> but we'll drop the missing values rows because removal of these very less rows will not impact statistical analysis</h4>
'''

# %%
for c in ['title', 'emp_title', 'emp_length', 'last_pymnt_d', 'revol_util']:
    filtered_df = filtered_df[~filtered_df[c].isnull()]
# %%

moreMissingCols = round(
    100*(filtered_df.isnull().sum()/len(filtered_df.index)), 2).sort_values(ascending=False)
moreMissingCols[moreMissingCols > 0]

# %% [markdown]
# ## We have Non null data now

# %%
filtered_df.to_csv('non_null_loans.csv', encoding='utf-8', index=False)

# %%
cleanDf: pd.DataFrame = pd.read_csv('non_null_loans.csv', encoding='utf-8')
# %% [markdown]
'''
**unique value counts**
'''
# %%
unique_value_counts = cleanDf.nunique().sort_values()
unique_value_counts
# %% [markdown]
'''
<h3>We could see that there are many columans which have single value, we must get rid of them as they are non informative</h3>
'''
# %%
unique_value_counts = cleanDf.nunique().sort_values()
colsToRemove = unique_value_counts[unique_value_counts == 1]
colsToRemove

# %%
cleanDf = cleanDf.drop(colsToRemove.index, axis=1)
# %%

cleanDf.nunique().sort_values()
# %% [markdown]
'''
<h3>Remove identity columns ie id, member_id, url</h3>
'''
# %%
cleanDf.drop(['id', 'member_id', 'url'], axis=1, inplace=True)

# %%
cleanDf.nunique().sort_values()

# %%
cleanDf.isnull().sum().sum()

# %%
cleanDf.to_csv('loans_with_required_columns.csv',
               encoding='utf-8', index=False)

# %%
cleanDf = pd.read_csv('loans_with_required_columns.csv', encoding='utf-8')
# %%
cleanDf.shape

# %%
cleanDf.head()
# %% [markdown]
'''
<h3>Investigating and correct Data Types</h3>
'''
# %%
cleanDf.info()
# %% [markdown]
# ### Converting month column to datetime

# %%
for c in ['issue_d', 'earliest_cr_line', 'last_pymnt_d']:
    cleanDf[c] = pd.to_datetime(cleanDf[c], format='%b-%y')

# %% [markdown]
'''
<h3>Fix the data types</h3>

'''
# %%
T = TypeVar('T')


def cleaner(value: str, func: Callable[[str], T]) -> T:
    if value is not None:
        value = str(value).strip()
        if len(value) == 0:
            return None
        return func(value)
    return None


term_cleaner: Callable[[str], int] = lambda term: cleaner(
    term, lambda term: int(term.split()[0]))


percent_leaner: Callable[[str], float] = lambda value: cleaner(
    value, lambda int_rate: float(int_rate.strip().replace("%", '')))


def emp_length_c(emp_length: str) -> int:
    emp_length = str(emp_length)
    if emp_length.lower() == 'nan':
        return None
    if emp_length.startswith('10+'):
        return 99
    if emp_length.startswith('<'):
        return 0
    return int(emp_length.split()[0])


emp_length_cleaner: Callable[[str],
                             int] = lambda value: cleaner(value, emp_length_c)


zip_code_cleaner: Callable[[str], str] = lambda value: cleaner(
    value, lambda zip: int(value[:3]))
# %%

cleanDf['term'] = cleanDf['term'].apply(term_cleaner)
# %%
cleanDf['int_rate'] = cleanDf['int_rate'].apply(percent_leaner)
# %%
cleanDf['emp_length'] = cleanDf['emp_length'].apply(emp_length_cleaner)
# %%
cleanDf['zip_code'] = cleanDf['zip_code'].apply(zip_code_cleaner)
# %%
cleanDf['revol_util'] = cleanDf['revol_util'].apply(percent_leaner)

# %%
cleanDf.info()
# %%

cols_to_string_normalization = ['grade', 'sub_grade', 'emp_title',
                                'home_ownership', 'verification_status', 'loan_status', 'purpose', 'title', 'zip_code', 'addr_state']

# %%


def normalize_string_values():
    for c in cols_to_string_normalization:
        cleanDf[c] = cleanDf[c].astype(str).str.upper()

# %%


normalize_string_values()

# %%
cols_to_be_categorical = cols_to_string_normalization + ['term', 'emp_length']
# %%


def normalize_categorical_values():
    for c in cols_to_string_normalization:
        cleanDf[c] = cleanDf[c].astype('category')

# %%


normalize_categorical_values()
# %%
cleanDf.info()

# %% [markdown]
'''
**Remove duplicate**
'''
# %%

cleanDf = cleanDf.drop_duplicates()
# %% [markdown]
'''
<h2>Uni-variate analysis</h2>
'''
# %%


class PlotType(Enum):
    BAR = auto()
    HIST = auto()
    BOX = auto()
    BOXEN = auto()
    VIOLIN = auto()

# %%


def plot_univariate_series(series: pd.Series, title: str, xlabel: str, ylabel: str, display_format: str = '{0:,.0f}', x_rotation=0, y_rotation=0, figsize=None, show_count=True, plot_type: PlotType = PlotType.BAR, bins: int = 20) -> None:
    """Plots bar/hist plot for series using seaborn library

    Args:
        series (pd.Series): series data
        title (str): plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        display_format (str, optional): values on top of each bar. Defaults to '{0:,.0f}'.
        x_rotation (int, optional): x-axis text rotation. Defaults to 0.
        y_rotation (int, optional): y-axis text rotation. Defaults to 0.
        figsize ([type], optional): figure size. Defaults to None.
        show_count (bool, optional): show values on bar. Defaults to True.
        plot_type (PlotType, optional): bar plot or histogram. Defaults to PlotType.BAR.
        bins (int, optional): number of bins if plot is histogram. Defaults to None.
    """
    ax = None
    if figsize is not None:
        plt.figure(figsize=figsize)
    if plot_type == plot_type.BAR:
        ax = sns.barplot(x=series.index, y=series)
    if plot_type == plot_type.HIST:
        ax = sns.histplot(series, bins=bins)

    if plot_type == plot_type.BOX:
        ax = sns.boxplot(series)

    if plot_type == plot_type.BOXEN:
        ax = sns.boxenplot(series)

    if plot_type == plot_type.VIOLIN:
        ax = sns.violinplot(series)

    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)
    plt.title(title, size=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_count and ax is not None:
        # Ref: https://github.com/mwaskom/seaborn/issues/1582
        for i, p in enumerate(ax.patches):
            ax.annotate(display_format.format(series.iloc[i]), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 4), textcoords='offset points')
    plt.show()

# %%


def get_title(v): return v.replace("_", " ").title()


def get_univariate_cat_plot_strs(value: str) -> Tuple[str, str, str]:
    title_case = get_title(value)
    count_str = title_case + " Count"
    return count_str + " Plot", title_case, count_str


def get_univariate_hist_plot_strs(value: str) -> Tuple[str, str, str]:
    title_case = get_title(value)
    return title_case + " Distribution (Histogram)", title_case


def get_univariate_box_plot_strs(value: str) -> Tuple[str, str, str]:
    title_case = get_title(value)
    return title_case + " Box Plot", title_case
# %%


def plot_univariate_categorical_columns():
    for c in cleanDf.select_dtypes('category').columns:
        value_counts_ser = cleanDf[c].value_counts().sort_index()
        cnt_len = len(value_counts_ser)
        # display(cnt_len)
        if cnt_len < 16:
            t, xl, yl = get_univariate_cat_plot_strs(c)
            plot_univariate_series(value_counts_ser, t,
                                   xl, yl, figsize=(15, 10), x_rotation=45)


plot_univariate_categorical_columns()

# %%
num_cols = cleanDf.select_dtypes([np.number]).columns
num_cols
# %%


def describe_num_columns():
    for c in num_cols:
        display(f"Decribing {get_title(c)}")
        display(cleanDf[c].describe())


# %%
describe_num_columns()

# %%


def plot_univariate_continuous_columns_hist():
    for c in num_cols:
        t, xl = get_univariate_hist_plot_strs(c)
        plot_univariate_series(cleanDf[c], t,
                               xl, None, figsize=(15, 10), x_rotation=45, plot_type=PlotType.HIST, show_count=False)


# %%
plot_univariate_continuous_columns_hist()
# %%


def plot_univariate_continuous_columns_box():
    for c in num_cols:
        t, xl = get_univariate_box_plot_strs(c)
        plot_univariate_series(cleanDf[c], t,
                               xl, None, figsize=(15, 10), x_rotation=45, plot_type=PlotType.BOX, show_count=False)


# %%
plot_univariate_continuous_columns_box()

# %% [markdown]
'''
<h2> Bi-Variate analysis </h2>
'''
# %% [markdown]
'''
<h3> We are going to follow 5 bin policy ['Very Low', 'Low', 'Medium', 'High', 'Very High'] for any numerical reason because this is just natural and psychologically easy to adapting to people</h3>

**Unique Value counts in continuous columns**
'''

# %%

numeric_values_unique_count = pd.Series(
    [cleanDf[c].nunique() for c in num_cols], index=num_cols).sort_values()
cols_to_bin = numeric_values_unique_count[numeric_values_unique_count > 5].index

bin_cats = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
numeric_cat_cols = []


def make_cat_cols_for_numerics():
    for col in cols_to_bin:
        new_col_name = col+"_cat"
        numeric_cat_cols.append(new_col_name)
        cleanDf[new_col_name] = pd.cut(cleanDf[col], 5, labels=bin_cats)


# %%
make_cat_cols_for_numerics()

# %%
cleanDf.head()

# %%
cleanDf['loan_status'] = cleanDf['loan_status'].astype(int)
# %%
t_col = 'loan_status'

# %%
all_cat_columns = cleanDf.select_dtypes('category').columns
feature_info_gain = []
cols_under_observation = []
stats_groups = {}

# %%


def calculate_stats_for_columns():
    for var_col in all_cat_columns:
        v_counts = len(cleanDf[var_col].value_counts())
        if v_counts < 10:
            groups = cleanDf.groupby(var_col).agg(
                {t_col: ['mean', 'count', 'sum']}).reset_index()
            groups.columns = [var_col, "def_frac", 'total', 'def_count']
            groups['non_def'] = groups['total'] - groups['def_count']
            groups['non_def_frac'] = 1 - groups['def_frac']
            information_gain = groups['def_frac'].max(
            ) - groups['def_frac'].min()
            groups.sort_values(by='def_count', ascending=False, inplace=True)
            feature_info_gain.append(information_gain)
            stats_groups.update({var_col: groups})
            cols_under_observation.append(var_col)


# %%
calculate_stats_for_columns()

# %%
info_gain_ser = pd.Series(
    feature_info_gain, index=cols_under_observation).sort_values(ascending=False)
info_gain_ser


# %% [markdown]
'''
<h3>Plots </h3>
 1. Observations/comments need to be filled<br>
 2. A ploting function needs to be written to plot values from "stats_groups"
'''

# %%


def plot_information_gain_stats():
    for k in stats_groups:
        clean_k = k.replace('_cat', '') if k.endswith("_cat") else k
        title_case = get_title(clean_k)
        count_title = title_case + ": Default vs NonDefaulter Count"
        percent_title = title_case + ": Default vs NonDefaulter Percent"
        v1: pd.DataFrame = stats_groups[k][[k, 'def_count', 'non_def']]
        v1.columns = [title_case, "Defaulter Count", "Non-Defaulter Count"]
        v1 = v1.melt(id_vars=title_case)
        v1.columns = [title_case, 'For', "Count"]
        # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
        # https://medium.com/@yoonho0715/seaborn-factor-plot-params-2a3ed9cf71bc
        f = sns.factorplot(x=title_case, y='Count',
                           hue='For', data=v1, kind='bar')
        f.fig.suptitle(count_title, fontsize=12)
        # f.fig.set_size_inches((10, 10))

        v1: pd.DataFrame = stats_groups[k][[k, 'def_frac', 'non_def_frac']]
        v1['def_frac'] = (v1['def_frac']*100).round(2)
        v1['non_def_frac'] = (v1['non_def_frac']*100).round(2)
        v1.columns = [title_case, "Defaulter Percent", "Non-Defaulter Percent"]
        v1 = v1.melt(id_vars=title_case)
        v1.columns = [title_case, 'For', "Percent"]
        f = sns.factorplot(x=title_case, y='Percent',
                           hue='For', data=v1, kind='bar')
        f.fig.suptitle(percent_title, fontsize=12)
        # f.fig.set_size_inches((10, 10))

        plt.show()


plot_information_gain_stats()

# %%
info_gain_ser

# %%
