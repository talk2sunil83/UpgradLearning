# %%
import pandas as pd
from math import radians, sqrt, ceil
# %%

n = 10


def fib(n: int) -> None:
    if n <= 0:
        raise ValueError("n cannot be negative")
    elif n == 1:
        print(0)
    elif n == 2:
        print(0)
        print(1)
    else:
        pre_pre, pre = 0, 1
        print(pre_pre)
        print(pre)
        for i in range(n-2):
            next_v = pre_pre+pre
            print(next_v)
            pre_pre, pre = pre, next_v


fib(n)
# %%
n = 10


def is_prime(n: int) -> str:
    if n > 1:
        for i in range(2, ceil(sqrt(n))):
            if n % i == 0:
                return "number entered is not prime"
        return "number entered is prime"
    else:
        raise ValueError("n must be greater than one")


is_prime(n)

# %%

n = 152


def is_armstrong(n: int) -> bool:
    return sum([int(d)**3 for d in str(n)]) == n


print(is_armstrong(n))
# %%
n = 153


def f(n: int) -> str:
    pass


print(f(n))

# %%

df = pd.read_csv(
    "https://media-doselect.s3.amazonaws.com/generic/X0kvr3wEYXRzONE5W37xWWYYA/test.csv")

to_omit = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Embarked']
# %%

# cols_to_select = sorted(set(df.columns) - set(to_omit))
# cols_to_select

# print(df[sorted(set(df.columns) - set(to_omit))].head())

new_df = df.drop(to_omit, axis=1)
new_df = new_df[sorted(new_df.columns)]
print(new_df.head(5))
# %%
df = df[df.columns[~df.columns.isin(to_omit)]]
print(df.loc[:, sorted(list(df.columns))].head())
# %%
input_list = [[1, 2, 3, 4, 5, 6, 7], [1, 3, 7]]
series1 = pd.Series(input_list[0])
series2 = pd.Series(input_list[1])
out_list = series1[series1.isin(series2)].index  # store your output here
# do not alter this step, list must be int type for evaluation purposes
print(list(map(int, out_list)))
# %%
df = pd.read_csv(
    "https://media-doselect.s3.amazonaws.com/generic/8NMooe4G0ENEe8z9q5ZvaZA7/googleplaystore.csv")
# %%
df.shape
# %%
df.columns
# %%
df['Installs'].unique()
# %%
df = df[df['Installs'] != 'Free']
df['Installs'] = df['Installs'].apply(
    lambda ins: int(ins.replace("+", "").replace(",", "")))
print(df.corr())
# %%
print(df.corr())
# %%
"I am now a master of Logistic regression".title()
# %%
input_list = [7, 2, 0, 9, -1, 8]
# input_list = [6, 6, 6, 6, 6]
# input_list = [3, 1, 4, 4, 5, 5, 5, 0, 2, 2]


def get_second_largest(values):
    if values is not None and len(values) > 2:
        values = set(values)
        if len(values) == 1:
            return "not present"

        values = sorted(values, reverse=True)
        return values[1] if values[0] > values[1] else "not present"
    else:
        raise ValueError("list must have at least two elements")


print(get_second_largest(input_list))

# %%
input_list = [7, 2, 0, 9, -1, 8]
input_list = [6, 6, 6, 6, 6]
input_list = [3, 1, 4, 4, 5, 5, 5, 0, 2, 2]
input_list = [7, 7]


def get_second_largest(values):
    if values is not None and len(values) >= 2:
        values = set(values)
        if len(values) == 1:
            return "not present"
        return sorted(values, reverse=True)[1]
    else:
        raise ValueError("list must have at least two elements")


print(get_second_largest(input_list))
# %%
wholesale = pd.read_csv(
    'https://media-doselect.s3.amazonaws.com/generic/OkbnaOBqrBXZOpRQw1JGMgaM9/Wholesale_Data.csv')
wholesale.columns
# %%
wholesale['Channel'].unique()
# %%
#  'Hotel', 'Restaurant', and 'Cafe'.


# wholesale['Channel'] = wholesale['Channel'].replace({
#     "Hot":"Hotel",
#     "H":"Hotel"
#     "Hote"
# })

wholesale['Channel'] = wholesale['Channel'].apply(lambda c: "Hotel" if c.lower(
).startswith("h") else "Restaurant" if c.lower().startswith("r") else "Cafe")
wholesale['Channel'].unique()
# %%
wholesale = pd.read_csv(
    'https://media-doselect.s3.amazonaws.com/generic/OqwpypRKN09x5GYej2LvVrprn/Wholesale_Data_Cleaned.csv')
# %%
print(list(wholesale.groupby("Channel").sum().sum(
    axis=1).nsmallest(1).index)[0])
# %%
