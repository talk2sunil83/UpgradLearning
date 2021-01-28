# %%
import csv
from operator import index
from typing import Callable, TypeVar, Generic
from matplotlib.pyplot import annotate
import numpy as np
from numpy.__config__ import show
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
import os
%matplotlib inline

# %%
default_ecoding = 'iso-8859-1'
dollar_formatting = '${0:,.2f}'
float_formatting = '{0:,.2f}'
int_formatting = '{0:,.0f}'

# %%
df = pd.read_csv("loan.csv", encoding=default_ecoding)
# %%
pd.options.display.max_columns = 200
# %%
non_unique_cols = []
for c in df.columns:
    unique_values = df[c].unique()
    if len(unique_values) > 1:
        non_unique_cols.append(c)
        print(c)
# %%
len(non_unique_cols)
# %%
df1 = df[non_unique_cols]
# %%
df1.head()

# %%
rec_count = df1.shape[0]
# %%
null_counts = df1.isnull().sum() > 0

# %%
cols_with_nulls = null_counts[null_counts].index
# %%
df1[cols_with_nulls].isnull().sum().sort_values(ascending=False)
# %%
null_percent = round(
    100*(df1[cols_with_nulls].isnull().sum()/rec_count), 2).sort_values(ascending=False)
null_percent
# %%
df1['next_pymnt_d'].value_counts()

# %%
df1.shape


def write_csv(data: pd.DataFrame, file_name: str,
              quoting: int = 1, encoding: str = 'utf-8') -> None:
    data.to_csv(file_name, index=False, quoting=quoting, encoding=encoding)


def write_excel(data: pd.DataFrame, file_name: str, encoding: str = 'utf-8', sheet_name: str = None, index=False) -> None:
    try:
        book = load_workbook(file_name)
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            data.to_excel(writer, sheet_name=sheet_name,
                          encoding=encoding, index=index)
            writer.save()
    except FileNotFoundError as e:
        data.to_excel(file_name, sheet_name=sheet_name,
                      encoding=encoding, index=index)


# %%
df1.drop('url', axis=1, inplace=True)
df1.drop('collections_12_mths_ex_med', axis=1, inplace=True)
df1.drop('chargeoff_within_12_mths', axis=1, inplace=True)
df1.drop('tax_liens', axis=1, inplace=True)
write_csv(df1, 'loan_with_non_null.csv')
# %%
data_dict_clean_file = "Data_Dictionary_Clean.xlsx"
if not os.path.exists(data_dict_clean_file):
    data_dict_file = 'Data_Dictionary.xlsx'

    sheets = ['LoanStats', 'RejectStats']
    columns = ["LoanStatNew", "RejectStats File"]
    for sheet, column in zip(sheets, columns):
        data_dict = pd.read_excel(data_dict_file, sheet_name=sheet)
        t = data_dict[data_dict[column].isin(df1.columns)]
        write_excel(t, data_dict_clean_file, sheet_name=sheet)

# %%
df = df1
df.head()
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


zip_code_cleaner: Callable[[str], int] = lambda value: cleaner(
    value, lambda zip: int(value[:3]))

# %%


# %%
df['term'] = df['term'].apply(term_cleaner)
# %%
df['int_rate'] = df['int_rate'].apply(percent_leaner)
# %%
df['emp_length'] = df['emp_length'].apply(emp_length_cleaner)
# %%
df['zip_code'] = df['zip_code'].apply(zip_code_cleaner)
# %%
df['revol_util'] = df['revol_util'].apply(percent_leaner)

# %%
write_csv(df, 'loan_with_non_null.csv')

# %%
# %%
null_counts = df.isnull().sum() > 0
rec_count = df.shape[0]
cols_with_nulls = null_counts[null_counts].index
null_percent = round(
    100*(df[cols_with_nulls].isnull().sum()/rec_count), 2).sort_values(ascending=False)
null_percent
# %%
df.nunique().sort_values()
# %%
sns.heatmap(df[['pub_rec_bankruptcies', 'pub_rec', 'loan_status']].corr())
# %%
df.groupby('loan_status')['pub_rec_bankruptcies'].count()/df.shape[0]
# %%
round(100*(df.groupby('loan_status')
           ['pub_rec_bankruptcies'].count()/df.shape[0]), 2)
# %%
