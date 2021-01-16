# %% [markdown]
'''
<h1>BoomBikes Data</h1>
<h2>Abstract</h2>
<p>A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.</p> 
<p>
<b>BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19.</b> They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits

<h2>Problem Statement</h2>
<p>BoomBikes has contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:</p> 

 - Which variables are significant in predicting the demand for shared bikes.
 - How well those variables describe the bike demands
 
<p>Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors.</p> 

<h2>Business Goal</h2>

<p>We are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. </p> 


<h2>Solution Approach</h2>
<b>We'll use multiple leinear regression to solve this problem</b>


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
 
<h2>Assumptions</h2>
 
 1. As business is looking for strategy for future time will be crucial factor<br>
 2. <b>Will be using plotly library to keep plots clean and interactive</b>
'''
# %% [markdown]
'''
Libs Installations
pip install numpy pandas matplotlib seaborn plotly chart-studio statsmodels scikit-learn pingouin --no-cache-dir
'''
# %% [markdown]
'''
**References**

https://plotly.com/python/distplot/  
https://plotly.com/python/plotly-express/
'''
# %% [markdown]
'''
<h2>Imports</h2>
'''
# %%

# Util Libs
# %%
# Data Reading, Wrangling, and processing Libs

# Plotting Libs
# Modeling Libs


# Feature Selection


# Statistical Libs

# %% [markdown]
'''
<h2>Setup</h2>
'''
# %%
# Set current library for as working dir
from IPython.display import Markdown
import plotly.figure_factory as ff
from IPython.display import display
import calendar
from typing import Callable, Dict, Sequence, Tuple, Union
from enum import Enum, auto
import os
import warnings
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
os.chdir(os.path.dirname(__file__))

# Pandas Setup
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.mode.use_inf_as_na = True
pd.set_option('precision', 8)
pd.options.display.float_format = '{:.4f}'.format

# Magic Commands
# : matplotlib
%matplotlib inline
# Seaborn Style
sns.set(style="whitegrid")

# Ignore Warnings
warnings.filterwarnings('ignore')

# Jupyter setup
init_notebook_mode(connected=True)
# %% [markdown]
'''
<h2>Global variables</h2>
'''
# %%
# columns
instant = 'instant'
dteday = 'dteday'
season = 'season'
yr = 'yr'
mnth = 'mnth'
holiday = 'holiday'
weekday = 'weekday'
workingday = 'workingday'
weathersit = 'weathersit'
temp = 'temp'
atemp = 'atemp'
hum = 'hum'
windspeed = 'windspeed'
casual = 'casual'
registered = 'registered'
cnt = 'cnt'
month = 'month'
date_of_month = 'date_of_month'
week_day = 'week_day'
week_day_name = 'week_day_name'
quarter = 'quarter'
week_of_year = "week_of_year"
numeric_columns = [temp, atemp, hum, windspeed, casual, registered, cnt]

col_name_map = {
    dteday: "Date On",
    season: "Season",
    yr: "Year",
    mnth: "Month Number",
    holiday: " Is holiday",
    weekday: "Weekday",
    workingday: "Is Working Day",
    weathersit: "Weather Type",
    temp: "Temperature in Celsius",
    atemp: "Feeling temperature in Celsius",
    hum: "Humidity in %",
    windspeed: "Wind Speed",
    casual: "Count of casual users",
    registered: "Count of registered users",
    cnt: "Total User Counts",
    month: "Month Of Year",
    date_of_month: "Date of Month",
    week_day: "Week Day",
    week_day_name: "Week day Name",
    quarter: "Quarter",
    week_of_year: "Week of the year"
}


seasons_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
weathersit_map = {
    1: "Clear, Few clouds, Partly cloudy, Partly cloudy",
    2: "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
    3: "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
    4: "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"
}
workingday_map = {1: "Working", 0: "Non Workind"}
#  Set name value mapping
month_name_map = {i: date(datetime.now().year, i, 1).strftime('%B')
                  for i in range(1, 13)}
holiday_map = {1: "Holiday", 0: "Non Holiday"}
# Listing from Tuesday because in this file Tuesday is considered as 0
days_of_week = ["Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday", "Monday"]
weekday_map = {i: days_of_week[i] for i in range(len(days_of_week))}
year_map = {0: 2018, 1: 2019}
date_of_month_map = {i: i for i in range(1, 32)}
quarter_map = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
col_value_map = {
    workingday: workingday_map,
    month: month_name_map,
    mnth: month_name_map,
    holiday: holiday_map,
    week_day: weekday_map,
    weekday: weekday_map,
    yr: year_map,
    season: seasons_map,
    weathersit: weathersit_map,
    date_of_month: date_of_month_map,
    quarter: quarter_map
}
# %% [markdown]
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
        value_counts_ser = dataframe[c].value_counts().sort_index()
        cnt_len = len(value_counts_ser)
        if cnt_len < 16:
            t, xl, yl = get_univariate_cat_plot_strs(c)
            value_counts_ser.index = value_counts_ser.index.map(
                col_value_map[c])
            plot_univariate_series(value_counts_ser, t,
                                   xl, yl, **kwargs)


# %% [markdown]
'''<h2>Step 1: Reading and Understanding the Data</h2>'''
# %%
df = pd.read_csv("day.csv")
# %%
df.shape
# %%
df.head()

# %%
df.describe().T

# %%
df.info()

# %%
df.columns

# %% [markdown]
'''
<h3>Dropping Identity</h3>
'''
# %%

df.drop(instant, axis=1, inplace=True)

# %% [markdown]
'''
**We can observe that**

 - No Nulls in data
 - dteday should be date
 - season, yr, mnth, holiday,weekday, workingday, weathersit should be category

**So we'll convert them correspondingly**
'''
# %%
# convert columns to category
cat_columns = [season, yr, mnth, holiday, weekday, workingday, weathersit]
df[cat_columns] = df[cat_columns].astype('category')
df.dtypes
# %%
# convert dteday as date
df[dteday] = pd.to_datetime(df[dteday], format='%d-%m-%Y')
df.dtypes


# %% [markdown]
'''
**Data types look good now**
'''
# %%
df.describe().T

# %% [markdown]
'''<h2>EDA</h2>'''

# %% [markdown]
'''
<h2>Observation about target columns</h2>
<h4>
Here we can select casual, registered, and cnt as target variables separately for analysis

 - Though we'll see their relationship
 - For this analysis we'll consider **cnt as target column**
</h4>
'''

# %% [markdown]
'''<h2>Step 2. Visualizing the Data</h2>'''
# %% [markdown]
'''
<h3>Uni Variate analysis</h3>
'''

# %%
# Enable
# cat_columns = df.select_dtypes(include='category').columns
# plot_univariate_categorical_columns(cat_columns, df)

# %% [markdown]
'''
**Conclusions:**

 1. Seasons are almost equally distributed
 2. We have data for complete two years
 3. Data is present for all months and all days
 4. There were 21 days for holiday
 5. 231 were non working and 499 working days
 6. Weathers:
   - 463 days were clear, Few clouds, Partly cloudy, Partly cloudy
   - 246 days were Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
   - 21 days were Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
 7. Mostly weather was good
'''
# %%


def plot_hist(df: pd.DataFrame, x: str, y: str, title: str, **kwargs):
    """Plot Histograms for given dataframe and selected columns

    Args:
        df (pd.DataFrame): Dataframe
        x (str): x-axis column
        y (str): y-axis column
        title (str): Title for graph
    """
    fig = px.histogram(
        df, x=x, y=y, title="Histogram for " + title.title(), **kwargs)
    fig.show()


def plot_area(df: pd.DataFrame, x: str, y: str, title: str, **kwargs):
    """Plot AreaGraph for given dataframe and selected columns

    Args:
        df (pd.DataFrame): Dataframe
        x (str): x-axis column
        y (str): y-axis column
        title (str): Title for graph
    """
    fig = px.area(df, x=x, y=y, title="Trend for " +
                  title.title(), line_shape="spline", **kwargs)
    fig.show()


def plot_box(df: pd.DataFrame, x: str, y: str, title: str, **kwargs):
    """Plot BoxPlot for given dataframe and selected columns

    Args:
        df (pd.DataFrame): Dataframe
        x (str): x-axis column
        y (str): y-axis column
        title (str): Title for graph
    """
    fig = px.box(df, x=x, y=y, title="Box and Whisker plot for " +
                 title.title(), **kwargs)
    fig.show()


def plot_numeric_col(df: pd.DataFrame, x: str, y: str, title: str, plotters: Callable[[pd.DataFrame, str, str, str, dict], None], **kwargs) -> None:
    """This is a higher order function, which plot selected(passed function refs in plotters variable) plots for given dataframe and selected columns

    Args:
        df (pd.DataFrame): Dataframe
        x (str): x-axis column
        y (str): y-axis column
        title (str): Title for graph
        plotters ([type]): plotting functions
    """
    if df is not None and x is not None and y is not None and title is not None and plotters is not None:
        for plt_fun in plotters:
            plt_fun(df, x, y, title, **kwargs)


# %%
numeric_col_names = [col_name_map.get(cn) for cn in numeric_columns]

# %%


def box_for_numerics() -> None:
    for col in numeric_columns:
        col_name = col_name_map.get(col)
        fig = px.box(df, y=col, title="Box and Whisker plot for " +
                     col_name.title(), labels={col: col_name})
        fig.show()


# %%
# box_for_numerics() # Enable
# %% [markdown]
'''
**It is clear that 2019 has been better than 2018 for BoomBike**
'''
# %% [markdown]
'''
Lets plot 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered' and 'cnt' plots against 'dteday'
'''

# %%


def plot_numeric_columns_trends_for_data() -> None:
    for col in numeric_columns:
        col_name = col_name_map.get(col)
        plot_numeric_col(df, x=dteday, y=col, title=col_name,
                         plotters=[plot_hist, plot_area], labels={dteday: "Dates", col: col_name})


# %%
# plot_numeric_columns_trends_for_data() # Enable
# %%


def numeric_values_box(df: pd.DataFrame, x_col_name: str) -> None:
    for col in numeric_columns:
        col_name = col_name_map.get(col)
        df_plot = df[[col, x_col_name]]
        df_plot[x_col_name] = df_plot[x_col_name].map(
            col_value_map.get(x_col_name))
        plot_numeric_col(df_plot, x=x_col_name, y=col, title=col_name,
                         plotters=[plot_box], labels={x_col_name: col_name_map.get(x_col_name), col: col_name}, color=x_col_name)


# %%
#  year wise change in numeric values
# numeric_values_box(df, yr) # Enable

# %%
# month wise change in numeric values
# numeric_values_box(df, mnth) # Enable

# %% [markdown]
'''
<h3>Date Break up features</h3>
'''

# %%
# Extract date parts
df[date_of_month] = df[dteday].dt.day
df[quarter] = df[dteday].dt.quarter
df[week_day_name] = df[dteday].dt.day_name
df[week_of_year] = df[dteday].dt.week
# %%
# Day wise
# numeric_values_box(df, date_of_month) # Enable
# %%
# Quarter wise
# numeric_values_box(df, quarter) # Enable
# %%
# Week days wise
# numeric_values_box(df, weekday)

# %% [markdown]
'''
<h3>Scatter plots </h3>
'''
# %%


def plot_scatter_col_vs_counts(data: pd.DataFrame, col: str, **kwargs) -> None:
    for cnt_col in [casual, registered, cnt]:
        plot_df = data[[col, cnt_col, yr]]
        plot_df[yr] = plot_df[yr].map(col_value_map.get(yr))
        fig = px.scatter(plot_df, x=col, y=cnt_col, color=yr, title="Scatter for " +
                         # size=cnt_col,
                         col_name_map.get(col) + " vs " + \
                         col_name_map.get(cnt_col),
                         labels={
                             yr: col_name_map.get(yr),
                             col: col_name_map.get(col),
                             cnt_col: col_name_map.get(cnt_col),
                         }, ** kwargs)
        fig.show()


# %%
# plot_scatter_col_vs_counts(df, temp)  # Enable


# %%
# User type Relationship

fig = px.line(df, x=dteday, y=[casual, registered, cnt], width=1200)
fig.show()

# %%


def plot_bike_usage_trends_vs_numeric_columns() -> None:
    for num_col in numeric_columns:
        fig = px.line(df, x=dteday, y=num_col, width=1200,
                      #   facet_col=dteday, facet_col_spacing=0.04,
                      labels={
                          num_col: col_name_map.get(num_col),
                          dteday: col_name_map.get(dteday),
                      }, title=f"{col_name_map.get(num_col)} Trend")
        fig.show()


# %%
# plot_bike_usage_trends_vs_numeric_columns() # Enable
# %%

# %% [markdown]
'''
<h2>Bi-Variate Analysis</h2>
'''
# %% [markdown]
'''
**Pair plot**
'''
# %%
# pair plot
# Enable
# fig = px.scatter_matrix(df[numeric_columns], width=1200, height=1200)
# fig.show()

# %%
# Numeric column heatmap
# Enable
# plt.figure(figsize=(8, 8))
# sns.heatmap(df[[temp, atemp, hum, windspeed, cnt]].corr(),
#             annot=True, cmap="YlGnBu")
# plt.show()
# %% [markdown]
'''
Growth check for each count ie casual, registered and cnt
'''
# %%
# Enable
# df_for_growth = df[[yr, casual, registered, cnt]].melt(
#     id_vars=[yr], value_vars=[casual, registered, cnt])
# plot_numeric_col(df_for_growth, x=yr, y='value', title='Bike use trend break up',
#                  plotters=[plot_box], color='variable')

# %% [markdown]
'''
**We can see that each component has growth**
'''

# %% [markdown]
'''
<h3>Multi Variate analysis</h3>
'''

# %%
# Pairwise correlation
pg.pairwise_corr(df, columns=[temp, atemp, hum,
                              windspeed, cnt], method='pearson')
# %% [markdown]
'''
**Here we can see that temp<=>atemp have very high correlation so lets drop one and see the changes**
'''
# %%
# Pairwise correlation without temp
pg.pairwise_corr(df, columns=[atemp, hum,
                              windspeed, cnt], method='pearson')
# %% [markdown]
'''
**This looks good**
'''
# %%
pg.linear_regression(df[[temp, atemp, hum, windspeed]], df[cnt])
# %% [markdown]
'''
p value is very high for temp so lets drop and see the change 
'''
# %%
pg.linear_regression(df[[atemp, hum, windspeed]], df[cnt])
# %% [markdown]
'''
**We see that all the variables looks good now**
'''

# %% [markdown]
'''<h2>Step 3. Data Preparation</h2>'''
# %% [markdown]
'''
<h3>Lets encode and include categorical columns as well</h3>
'''
# %%

# %%
# Create Dummies function


def create_dummies(data_frame: pd.DataFrame) -> pd.DataFrame:
    clean_df = pd.DataFrame()
    cat_cols = list(data_frame.select_dtypes(include=['category']).columns)
    cat_cols = cat_cols + [date_of_month, quarter, week_day_name]
    for cat_col in cat_cols:
        dummies = pd.get_dummies(data_frame[cat_col], drop_first=True)
        dummies.columns = [cat_col + "_" +
                           str(i) for i in range(dummies.shape[1])]
        clean_df = pd.concat([clean_df, dummies], axis=1)

    return pd.concat([data_frame.drop(cat_cols, axis=1), clean_df], axis=1)


# %%
# Create Dummies
frame_with_dummies = create_dummies(df)
frame_with_dummies.columns

# %% [markdown]
'''
**We'll drop below columns:**

 - dteday : We have plotted necessary graphs and extrated required information
 - temp : Highly correlated with a temperature
 - casual & registered : cnt is sum of both and we are concentrating on total use in this study
 - week_of_year : Only 2 years of data so hard to get any pattern
'''
# %%
# Drop non required columns
unscaled_frame = frame_with_dummies.drop(
    [dteday, temp, casual, registered, week_of_year], axis=1)
unscaled_frame.columns
# %% [markdown]
'''<h2>Step 4. Splitting the Data into Training and Testing Sets</h2>'''
# %%
# Create X and y
y = unscaled_frame.pop(cnt)
X = unscaled_frame

print(X.shape, y.shape)

# %%
# Spit train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %%
# Numeric Columns Min-Max Scaling


def scale_columns(train_data: pd.DataFrame, test_data: pd.DataFrame, cols_to_scale: Sequence[str]):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data[cols_to_scale] = scaler.fit_transform(
        train_data[cols_to_scale].values)
    test_data[cols_to_scale] = scaler.transform(
        test_data[cols_to_scale].values)
    return train_data, test_data


# %%
# Scale training and test datasets
X_train, X_test = scale_columns(X_train, X_test, [atemp, hum, windspeed])
display(X_train.head())
display(X_test.head())
# %% [markdown]
'''
**Not scaling y as we don't need it**<br>
Ref: https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
'''
# %% [markdown]
'''<h2>Step 5. Building model(s)</h2>'''
# %% [markdown]
'''
<h4>We can see that there are total 61 independent variables so we'll use RFE to bring them down to 20(No Specific reason to choose but it seems that will be manageble to single removal) </h4>
'''
# %%

#  Feature Selector


def feature_selector(feature_selector_obj, train_data: pd.DataFrame, train_target: pd.DataFrame):
    selector = feature_selector_obj.fit(train_data, train_target)
    display(selector.support_)
    print(f"Total selected features: {selector.support_.sum()}")
    print(
        f"Selected feature Names: {list(train_data.columns[selector.support_])}")
    display(selector.ranking_)
    return selector


# %%
# Feature selection based on train data, so that we can prevent information leaking while feature selection
rfe_selector = feature_selector(
    RFE(LinearRegression(), n_features_to_select=20, step=1), X_train, y_train)
# %% [markdown]
'''
We'll validate the same with RFECV as well
'''
# %%
feature_selector(RFECV(LinearRegression(), step=1, cv=5), X_train, y_train)
# %% [markdown]
'''
RFECV, selected 16 features, it proves that 20 was correct selection, let's investigate other features
'''
# %%
rfe_selected_features_frame = X_train[list(
    X_train.columns[rfe_selector.support_])]
# %% [markdown]
'''
<h3>Current column set</h3>

['atemp', 'hum', 'windspeed', 'season_0', 'season_1', 'season_2', 'yr_0',
       'mnth_1', 'mnth_5', 'mnth_7', 'mnth_9', 'mnth_10', 'holiday_0',
       'weekday_5', 'workingday_0', 'weathersit_0', 'weathersit_1',
       'date_of_month_15', 'date_of_month_18', 'date_of_month_27']
'''
# %%
# convert everything to float_format, so that we can analyse them
rfe_selected_features_frame = rfe_selected_features_frame.astype(float)
rfe_selected_features_frame.dtypes
# %%


def column_wise_analysis(col: str, plot_reg_line: bool = False):
    # Adding constant/bias to train set
    col_name = col_name_map.get(col, col)
    print(f"Analysing {col_name}\n")
    X_train_lm = sm.add_constant(X_train[[col]])
    # Create a fitted model
    lr = sm.OLS(y_train, X_train_lm).fit()
    print(f"Model Summary for {col_name}\n")
    display(lr.summary())

    #  Plotting Reg line for selected column
    if plot_reg_line:
        column_data = X_train_lm.iloc[:, 1]
        y_preds = lr.params["const"] + lr.params[col]*column_data
        print(f"\nPlotting Reg line for {col_name}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=column_data, y=y_train,
                                 mode='markers', name=col_name))
        fig.add_trace(go.Line(x=column_data, y=y_preds,
                              name=f"Reg line for {col_name}"))
        fig.show()


def all_columns_analysis(X: pd.DataFrame, y: pd.Series):
    # Adding constant/bias to train set
    X_train_lm = sm.add_constant(X)
    # Create a fitted model
    lr = sm.OLS(y, X_train_lm).fit()
    print(lr.summary())
    return lr

# Ref: Taken from


def get_vif(dataframe: pd.DataFrame):
    # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
    vif = pd.DataFrame()
    vif['Features'] = dataframe.columns
    vif['VIF'] = [variance_inflation_factor(
        dataframe.values, i) for i in range(dataframe.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    display(vif)


# %% [markdown]
'''
Lets see all features
'''
# %%
all_columns_analysis(X_train, y_train)
# %% [markdown]
'''
Though R-squared, Adj. R-squared, F-statistic looks good but many features have very high p-value
'''

# %% [markdown]
'''
Lets see for VIF
'''
# %%
get_vif(X_train)
# %% [markdown]
'''
Many columns have inf as VIF that means we can remove many columns
'''
# %% [markdown]
'''
<h3>We'll start removing column with minimal importance ie high p and high VIF first</h3>
'''
# %% [markdown]
'''
<h3>Iteration 1</h3>
'''
# %% [markdown]
all_columns_analysis(rfe_selected_features_frame, y_train)
# %% [markdown]
'''
R-squared and  Adj. R-squared are almost same even after removal of almost 2/3 of features
R-squared, Adj. R-squared, F-statistic looks good but p-value for holiday_0: 0.15, and date_of_month_18: 0.102 are high
'''
# %% [markdown]
'''
VIF
'''
# %%
get_vif(rfe_selected_features_frame)
# %% [markdown]
'''
VIF for 
atemp	27.4800
hum	19.4100
season_1	7.9800
workingday_0	5.0500

are significantly high but their p-values are zero ie they have significance 

So in this case I'll go with p-value and remove holiday_0 column and see the model behavior
'''

# %% [markdown]
'''
<h3>Iteration 2</h3>
'''
# %%
# drop holiday_0
itr2_df = rfe_selected_features_frame.drop('holiday_0', axis=1)
# %%
all_columns_analysis(itr2_df, y_train)
# %% [markdown]
'''
I can see that there is no significance drop in Adj. R-squared and R-squared is same - So choice was good
R-squared, Adj. R-squared, F-statistic looks good but p-value for date_of_month_18: 0.107 is high
'''
# %% [markdown]
'''
VIF
'''
# %%
get_vif(itr2_df)
# %% [markdown]
'''
VIF for 
atemp	27.4100 from 27.4800
hum	18.9800 from 19.4100
season_1	7.9600 from 7.9800
workingday_0 4.4600	fromm 5.0500 which is fifth now 

we'll continue on high p-valued colum removal and drop date_of_month_18   
'''

# %% [markdown]
'''
<h3>Iteration 3</h3>
'''
# %%
# drop holiday_0
itr3_df = itr2_df.drop('date_of_month_18', axis=1)
# %%
all_columns_analysis(itr3_df, y_train)
# %% [markdown]
'''
I can see that there is no significance drop in Adj. R-squared and R-squared is same - So choice was good
R-squared, Adj. R-squared, F-statistic looks good and all p-values looks good
'''
# %% [markdown]
'''
VIF
'''
# %%
get_vif(itr3_df)
# %% [markdown]
'''
VIF for 
atemp remain 27.4100
hum	 remain 18.9800
season_1 remain7.9600

Now remove atemp having highest VIF
'''

# %% [markdown]
'''
<h3>Iteration 4</h3>
'''
# %%
# drop holiday_0
itr4_df = itr3_df.drop('atemp', axis=1)
# %%
all_columns_analysis(itr4_df, y_train)
# %% [markdown]
'''
A very significant drop in R-squared, Adj. R-squared both
and mnth_5 p-value is shooted to 0.682
'''
# %% [markdown]
'''
VIF
'''
# %%
get_vif(itr4_df)
# %% [markdown]
'''
VIF for 
hum	 dropped from 18.9800 to 12.61 but still significantly high

Now remove mnth_5 having highest p-value
'''

# %% [markdown]
'''
<h3>Iteration 5</h3>
'''
# %%
# drop holiday_0
itr5_df = itr4_df.drop('mnth_5', axis=1)
# %%
all_columns_analysis(itr5_df, y_train)
# %% [markdown]
'''
No significant drop in R-squared, Adj. R-squared
'''
# %% [markdown]
'''
VIF
'''
# %%
get_vif(itr5_df)
# %% [markdown]
'''
VIF for 
hum	 dropped from 12.61 to 12.60 but still significantly high

Now remove hum having highest VIF
'''

# %% [markdown]
'''
<h3>Iteration 6</h3>
'''
# %%
# drop holiday_0
itr6_df = itr5_df.drop('hum', axis=1)
# %%
lr = all_columns_analysis(itr6_df, y_train)
# %% [markdown]
'''
No significant drop in R-squared, Adj. R-squared, and all variables are significant
'''
# %% [markdown]
'''
VIF
'''
# %%
get_vif(itr6_df)
# %% [markdown]
'''
All variables have VIF less than 5
'''

# %%

# %% [markdown]
'''<h2>Step 6. Residual Analysis of the train data</h2>'''
# %%
itr6_df = sm.add_constant(itr6_df)
y_train_pred = lr.predict(itr6_df)

# %%

fig = ff.create_distplot([(y_train - y_train_pred)],
                         ["Residual Error Distribution"])
fig.show()

# %% [markdown]
'''
We can see that error look almost normally distributed and mean is near to zero so model looks good
'''

# %% [markdown]
'''
Scatter for y_train and y_train pred
'''
# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_train, y=y_train_pred,
                         mode='markers', name="Train Errors"))
fig.show()
# %% [markdown]
'''
We can there is constant variance of errors
'''
# %% [markdown]


'''<h2>Step 7. Making Predictions Using the Final Model</h2>'''


# %% [markdown]
'''
<h3>Check on test data</h3>
'''
# %%

X_test = X_test[itr6_df.columns[1:]]
# %%
all_columns_analysis(X_test, y_test)
# %%
X_test = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test)


# %% [markdown]
'''<h2>Step 8. Model Evaluation</h2>'''

# %% [markdown]
'''
Error plot for test data
'''

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_test_pred,
                         mode='markers', name="Test Errors"))
fig.show()

# %% [markdown]
'''
Very similar pattern like train errors
'''

# %% [markdown]
'''<h2>Step 9. Conclusion</h2>'''

# %%
#  Model coefficients
lr.params

# %%
features = lr.params
final_str = [f"$ Total Predicted Users = {features.const}"]
features = features.drop("const")
for v, ftr in zip(features, features.index):
    final_str.append(
        f" {'+' if v>0 else '-' } {str(round(abs(v), 4))}  \\times  {ftr.replace('_', '')} ")

final_str = "".join(final_str) + "$"
display(Markdown(final_str))

# %%
