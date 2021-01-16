# %%
import pandas as pd


def get_adj_r_sq(r_sq, n, p):
    return 1 - ((1-r_sq) * (n-1) / (n-p-1))


# %%
print(get_adj_r_sq(0.7, 100, 10))

# %%


def get_ivf_from_r_sq(r_sq: float) -> float:
    return 1 / (1-r_sq)


get_ivf_from_r_sq(0.9)
# %%
# %%
# Ref: https://condor.depaul.edu/sjost/it223/documents/correlation.htm


def get_ivf_from_corr(correlation: float) -> float:
    return 1 / (1-correlation**2)


get_ivf_from_corr(0.9)
# %%
# %%
input_list = [['Reetesh', 'Shruti', 'Kaustubh', 'Vikas', 'Mahima', 'Akshay'], ['No', 'Maybe', 'yes', 'Yes', 'maybe', 'Yes']]
name = input_list[0]
response = input_list[1]
df = pd.DataFrame({'Name': name, 'Response': response})

# Yes’, ‘No’, and ‘Maybe’. Write a code to map these variables to the values ‘1.0’, ‘0.0’, and ‘0.5’.
# %%
df['Response'] = df['Response'].str.lower()
# %%
df['Response'] = df['Response'].map({"yes": 1.0, "no": 0.0, "maybe": 0.5})

# %%
df.head()
# %%
