# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from IPython.display import display
%matplotlib inline

# %%
# df = pd.read_clipboard()
# df.to_csv("WOE pattern of the granular groups - fine classing.csv",  index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
# %%
# df = pd.read_clipboard()
# df.to_csv("WOE pattern of the groups post binning - coarse classing.csv",  index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
# %%

df_fine_classing = pd.read_csv(
    "WOE pattern of the granular groups - fine classing.csv")
df_coarse_classing = pd.read_csv(
    "WOE pattern of the groups post binning - coarse classing.csv")

# %%
df_fine_classing.head()


# %%
df_fine_classing.columns
# %%
df_coarse_classing.columns

# %%


def fill_WOE_and_IV(df: pd.DataFrame, target_file_name: str) -> None:
    total_good = df['No of good'].sum()
    total_bad = df['No of bad'].sum()
    df['WOE'] = np.log(df['No of good']/total_good) - \
        np.log(df['No of bad']/total_bad)
    df['Information Values'] = df['WOE']*(
        (df['No of good']/total_good) - (df['No of bad']/total_bad)
    )
    df.to_csv(target_file_name,
              index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    display(df)


# %%
fill_WOE_and_IV(df_fine_classing,
                "WOE pattern of the granular groups - fine classing.csv")
# %%
fill_WOE_and_IV(df_coarse_classing,
                "WOE pattern of the groups post binning - coarse classing.csv")
# %%
# %%
# df = pd.read_clipboard()

# df.to_csv("Woe values for categorical variable - Contract.csv",  index=False, encoding='utf-8')

df_fine_classing = pd.read_csv(
    "Woe values for categorical variable - Contract.csv")

fill_WOE_and_IV(df_fine_classing,
                "Woe values for categorical variable - Contract.csv")

# %%
file_name = "WOE pattern of categorical Variable - Employment Length.csv"
df = pd.read_clipboard()

df.to_csv(file_name,  index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

df_fine_classing = pd.read_csv(file_name)

fill_WOE_and_IV(df_fine_classing, file_name)
# %%
file_name = "Grade_woe_data_1.csv"
df = pd.read_clipboard()
df.to_csv(file_name,  index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
df = pd.read_csv(file_name)
fill_WOE_and_IV(df, file_name)

# %%
