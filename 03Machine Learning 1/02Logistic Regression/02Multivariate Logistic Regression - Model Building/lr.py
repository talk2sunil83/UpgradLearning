# %%
from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# %%
churn_data = pd.read_csv("churn_data.csv")
customer_data = pd.read_csv("customer_data.csv")
internet_data = pd.read_csv("internet_data.csv")

# %%
display(churn_data.shape)
display(customer_data.shape)
display(internet_data.shape)
# %%
display(churn_data.head())
display(customer_data.head())
display(internet_data.head())
# %%
display(churn_data.isnull().sum())
display(customer_data.isnull().sum())
display(internet_data.isnull().sum())

# %%
display(churn_data.describe().T)
display(customer_data.describe().T)
display(internet_data.describe().T)
# %%
customerID = "customerID"
telecom = churn_data.merge(customer_data, on=customerID, how='inner').merge(
    internet_data, on=customerID, how='inner')
# %%
telecom.head()
# %%
telecom.shape
# %%
telecom.describe().T
# %%
telecom.info()
# %%
PhoneService = 'PhoneService'
PaperlessBilling = 'PaperlessBilling'
Churn = 'Churn'
Partner = 'Partner'
Dependents = 'Dependents'
# %%
varlist = [PhoneService, PaperlessBilling, Churn, Partner, Dependents]
telecom[varlist] = telecom[varlist].apply(lambda x: x.map({'Yes': 1, "No": 0}))

# %%
telecom.head()
# %%
telecom['OnlineBackup'].value_counts()
# %%
telecom['OnlineSecurity'].value_counts()
# %%
telecom['DeviceProtection'].value_counts()
# %%
