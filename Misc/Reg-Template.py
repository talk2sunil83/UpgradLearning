# %% [markdown]
'''
<h1><<Company>></h1>
<h2>Abstract</h2>


<h2>Problem Statement</h2>


<h2>Business Goal</h2>

<h2>Solution Approach</h2>
<h2>Solution Steps</h2>
<h3>We'll solve this problem in below mentioned multiple steps:</h3> 

 1. Reading and Understanding the Data
 2. Visualizing the Data
 3. Data Preparation
 4. Splitting the Data into Training and Testing Sets
 5. Building model(s)
 6. Residual Analysis of the train data
 7. Making Predictions Using the Final Model
 8. Model Evaluation
 9. Conclusion
'''
# %% [markdown]
'''
Libs Installations
pip install numpy pandas matplotlib seaborn plotly chart-studio statsmodels scikit-learn --no-cache-dir
'''
# %% [markdown]
'''
<h2>Imports</h2>
'''
# %%
# Data Reading, Wrangling, and processing Libs
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Modeling Libs
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Feature Selection
from sklearn.feature_selection import RFE

# Plotting Libs
import matplotlib.pyplot as plt 
import seaborn as sns


# Model validation Libs
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Util Libs
import os
import warnings


# %% [markdown]
'''
<h2>Setup</h2>
'''
# %%
# Set current library for as working dir
os.chdir(os.path.dirname(__file__))

# Pandas Setup
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.mode.use_inf_as_na = True
pd.set_option('precision', 8)
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.expand_frame_repr =  False

# Magic Commands
# : matplotlib
%matplotlib inline
# Seaborn Style
sns.set(style = "whitegrid")

# Ignore Warnings
warnings.filterwarnings('ignore')
# %% [markdown]
'''
<h2>Global variables</h2>
'''
# %%


# %% [markdown]
'''
<h2>Utility Functions</h2>
'''

# %% [markdown]
'''<h2>Step 1: Reading and Understanding the Data</h2>'''
# %%
df = pd.read_csv("day.csv")

# %% [markdown]
'''<h2>Step 2. Visualizing the Data</h2>'''
# %% [markdown]
'''<h2>Step 3. Data Preparation</h2>'''
# %% [markdown]
'''<h2>Step 4. Splitting the Data into Training and Testing Sets</h2>'''
# %% [markdown]
'''<h2>Step 5. Building model(s)</h2>'''
# %% [markdown]
'''<h2>Step 6. Residual Analysis of the train data</h2>'''
# %% [markdown]
'''<h2>Step 7. Making Predictions Using the Final Model</h2>'''
# %% [markdown]
'''<h2>Step 8. Model Evaluation</h2>'''
# %% [markdown]
'''<h2>Step 9. Conclusion</h2>'''


# %%

