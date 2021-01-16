# %%i
import numpy as np
import pandas as pd


# %%

customer = pd.read_csv('https://query.data.world/s/y9rxL9mGdP6AXPiDaIL4yYm6DsfTV2')

customer['Cust_id'] = customer['Cust_id'].apply(lambda c: int(c[5:]))
print(customer.head(10))

# %%
from pydataset import data

# %%
sleepstudy =data('sleepstudy')

# %%
sleepstudy.head()

# %%
sleepstudy.dtypes

# %%
sleepstudy['Reaction'].apply(lambda r: float("%.1f" % r))

# %%
import pandas as pd
rating = pd.read_csv('https://query.data.world/s/EX0EpmqwfA2UYGz1Xtd_zi4R0dQpog')

# %%
rating_update = rating.drop_duplicates()

print(rating.shape)
print(rating_update.shape)

# %%
import pandas as pd
df = pd.read_csv("popularity.csv")
df.head()


# %%
df.columns = [c.strip() for c in df.columns]
df.columns
# %%
df['num_keywords'].value_counts()

# %%
df['num_keywords'].value_counts().mean()

# %%
df['shares'].mean()

# %%
df['shares'].median()

# %%
# df['shares'].plot.bar()

# %%
df['shares'].describe()

# %%
df['shares'].quantile(0.78)

# %%
df['shares'].plot.box()

# %%
df['shares'].quantile(0.8)

# %%
df['shares'].quantile(0.95)

# %%
qdf = df[df['shares']<=df['shares'].quantile(0.95)]

# %%
shares_ser = qdf['shares']
shares_ser.mean()
shares_ser.std()
# %%
((df.shape[0] - qdf.shape[0])*100)/df.shape[0]

# %%
nas = pd.read_csv("EDA_nas.csv")
# %%
nas.columns
# %%
nas.groupby("Watch.TV")['Maths..'].mean()

# %%
census = pd.read_csv("EDA_census.csv")

# %%
census.head()

# %%
female_data = census[((census['AreaName'] == 'INDIA') & (census['AgeGroup']=='20-24') & (census['LivLoc']=='Total'))]

# %%
female_data['IlliterateFemales']/female_data['TotalFemales']

# %%
india_data = census[((census['AreaName'] == 'INDIA') & (census['LivLoc']=='Total'))]

# %%
population = india_data['TotalPersons'].sum()
population

# %%
census.columns

# %%
age_group_lit_rate = pd.DataFrame()
age_group_lit_rate['AgeGroup'] = india_data['AgeGroup']
age_group_lit_rate['LitRate'] = (india_data['LiteratePersons']/india_data['TotalPersons'])*100


# %%
age_group_lit_rate.shape

# %%
import matplotlib.pyplot as plt
%matplotlib inline

# %%
plt.figure(figsize=(20, 5))
plt.bar(height='LitRate', x='AgeGroup', data=age_group_lit_rate)
plt.show()

# %%
age_group_lit_rate

# %%
state_data = census[((census['AreaName'] != 'INDIA') & (census['LivLoc']=='Total'))]

# %%
state_data.columns

# %%
statewise = state_data.groupby('AreaName').agg({'LiteratePersons':'sum', 'TotalPersons':'sum' }).reset_index()

# %%
statewise.head()

# %%
statewise['LitRate'] = 100*(statewise['LiteratePersons'] /statewise['TotalPersons'] )
statewise.sort_values(by='LitRate')

# %%
import pandas as pd
df = pd.read_csv("EDA_Gold_Silver_prices.csv")
df.head()

# %%
df.corr(method='pearson')

# %%
from datetime import datetime as dt

df['date'] = df['Month'].apply(lambda input: dt.strptime(input, '%b-%y'))
df['date']

# %%
df_2008 = df[df['date'].dt.year==2008][['SilverPrice','GoldPrice']]
df_2008.corr(method='pearson')

# %%
df['year'] = df['date'].dt.year

# %%
import matplotlib.pyplot as plt
plt.scatter( data= df.groupby(by='year').corr().reset_index().iloc[0::2][['year', 'GoldPrice']], x='year', y='GoldPrice')

plt.show()
# %%
df = pd.read_csv("currencies.csv")

# %%
df.columns

# %%
df.dtypes

# %%
tdf = df[["Euro","Japanese Yen","U.K. Pound Sterling","U.S. Dollar","Australian Dollar","Indian Rupee"]]

# %%
tdf.dtypes

# %%
import seaborn as sns

# %%
plt.figure(figsize=(8,8))
sns.heatmap(data=tdf.corr(), annot=True)
plt.show()
# %%
import pandas as pd

# %%
df =  pd.read_csv('nas.csv')
df.columns

# %%
df.head()

# %%
df.dtypes

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
q1_data = df[['Mother.edu', 'Siblings']]

# %%
q1_data['Mother.edu'].unique()

# %%
q1_data[q1_data['Mother.edu']=='Illiterate']['Siblings'].mode()


# %%
df['Age'].unique()

# %%
q2_data =  df[df['Age']!='11- years']

# %%
q2_data = q2_data[['Father.edu', 'Age', 'Science..']]

# %%
q2_data.head()

# %%
q2_data = q2_data.groupby(by=['Father.edu', 'Age']).agg({'Science..':'mean'}).reset_index()

# %%
q2_data.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# plt.figure(figsize=(10,10))
sns.catplot(data=q2_data, x='Father.edu', y='Science..', kind='box')
sns.catplot(data=q2_data, x='Age', y='Science..', kind='box')
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# %%
df = pd.read_csv("nas.csv")

# %%
df.head()

# %%
df.shape

# %%
df.columns

# %%
import pandas as pd
cust_rating = pd.read_csv('https://query.data.world/s/ILc-P4llUraMaYN6N6Bdw7p6kUvHnj')
# %%
cust_rating.columns
# %%
cust_rating['avg_rating'] = round(
    (cust_rating['rating']+
    cust_rating['food_rating']+
    cust_rating['service_rating'])/3)

print(cust_rating.head(10))

# %%
import numpy as np
import pandas as pd
# %%
df = pd.read_csv("odi-batting.csv")

# %%
df.columns

# %%
df.dtypes

# %%
df.sample(20)

# %%
q2_data = df[df['Runs']>=100]

# %%
q2_data[['Player', 'Runs']].groupby("Player").count().reset_index().sort_values(by='Runs', ascending=False)

# %%
q2_data['sr'] = 100*(q2_data['Runs']/q2_data['Balls'])
q2_data.sort_values(by='sr', ascending=False)

# %%
q2_data['year'] = pd.to_datetime(q2_data['MatchDate']).dt.year

# %%
q2_data[q2_data['Country']=='India'].groupby('year').count().reset_index().sort_values(by='Runs', ascending=False)

# %%
import pandas as pd
order = pd.read_csv('https://query.data.world/s/3hIAtsCE7vYkPEL-O5DyWJAeS5Af-7')
order['Order_Date'] = pd.to_datetime(order['Order_Date'])

# %%
order.columns

# %%
order.dtypes

# %%
order.sample(20)
# %%
order.head(20)
# %%

order['day'] = order['Order_Date'].dt.day #Type your code here

print(order.head(10))

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# %%
df = pd.read_csv("grades.csv")


# %%
df.dtypes

# %%
df.head()

# %%
df['file_type'] = df['submission'].apply(lambda s: s.split(".")[-1].lower())

# %%
df['file_type']

# %%
df['role_number'] = df['submission'].apply(
    lambda s: s.split("/")[-1].split(".")[-2].lower())

# %%
df["submit_time_p"]= pd.to_datetime(df["submit_time"])

# %%
df["submit_day"]= df["submit_time_p"].dt.day
df["submit_hour"]= df["submit_time_p"].dt.hour

# %%
df.head()

# %%
df["submit_day"].describe()

# %%
df["submit_hour"].describe()

# %%
df["submit_hour"].plot.box()

# %%
df["submit_day"].plot.box()

# %%
t = df.groupby(by='file_type')['submission'].count().reset_index().sort_values(by='submission', ascending=False)

# %%
t['percent'] = 100*(t['submission']/ df.shape[0])

# %%
from datetime import datetime

# %%
q2_data = df[df['submit_time_p'] >= datetime(2017, 1, 3,23, 59, 59, 0)]
q2_data.shape
# %%
df['submit_date'] = df['submit_time_p'].dt.date
# %%
df.groupby(by='submit_date').count().reset_index().sort_values(by='submission', ascending=False)
# %%
t2 = df.groupby(by='submit_hour').count().reset_index().sort_values(by='submission', ascending=False)[['submit_hour','submission']]

# %%
sns.barplot(x='submit_hour', y='submission', data=t2)

# %%
