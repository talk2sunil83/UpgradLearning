echo 'export MFTA_ORACLE_CON_STR="{\"username\":\"MFTBC_RO\", \"password\":\"Ro_95#mfT##87\", \"dbdetails\":\"53.124.91.86:15521/MFTBCUAT2\"}"' >> ~/.bashrc
echo 'export MFTA_ORACLE_CON_STR="{\"username\":\"MFTBC_RO\", \"password\":\"Ro_95#mfT##87\", \"dbdetails\":\"53.124.91.86:15521/MFTBCUAT2\"}"' >> ~/.bash_profile


echo 'export SQLAZURECONNSTR_KPYODBCCONSTR="DRIVER={ODBC Driver 17 for SQL Server};SERVER=kubota-tmap-sql-server.database.windows.net;PORT=1433;DATABASE=KubotaTmapSqlServer;UID=kubotatmapsqladmin;PWD=Tavant123TMAP$"' >> ~/.bashrc
echo 'export SQLAZURECONNSTR_KSQLALCHEMYCONSTR="DRIVER={ODBC Driver 17 for SQL Server};SERVER=kubota-tmap-sql-server.database.windows.net,1433;DATABASE=KubotaTmapSqlServer;UID=kubotatmapsqladmin;PWD=Tavant123TMAP$"' >> ~/.bash_profile

echo 'export SQLAZURECONNSTR_KPYODBCCONSTR="DRIVER={ODBC Driver 17 for SQL Server};SERVER=kubota-tmap-sql-server.database.windows.net;PORT=1433;DATABASE=KubotaTmapSqlServer;UID=kubotatmapsqladmin;PWD=Tavant123TMAP$"' >> ~/.bashrc
echo 'export SQLAZURECONNSTR_KSQLALCHEMYCONSTR="DRIVER={ODBC Driver 17 for SQL Server};SERVER=kubota-tmap-sql-server.database.windows.net,1433;DATABASE=KubotaTmapSqlServer;UID=kubotatmapsqladmin;PWD=Tavant123TMAP$"' >> ~/.bash_profile

echo 'export PATH=${PATH}:${MFTA_ORACLE_CON_STR}:${SQLAZURECONNSTR_KPYODBCCONSTR}:${SQLAZURECONNSTR_KSQLALCHEMYCONSTR}' >> ~/.bash_profile
echo 'export PATH=${PATH}:${MFTA_ORACLE_CON_STR}:${SQLAZURECONNSTR_KPYODBCCONSTR}:${SQLAZURECONNSTR_KSQLALCHEMYCONSTR}' >> ~/.bashrc

-----------------------------------------------------------------
export MFTA_ORACLE_CON_STR="{\"username\":\"MFTBC_RO\", \"password\":\"Ro_95#mfT##87\", \"dbdetails\":\"53.124.91.86:15521/MFTBCUAT2\"}"
export SQLAZURECONNSTR_KPYODBCCONSTR="DRIVER={ODBC Driver 17 for SQL Server};SERVER=kubota-tmap-sql-server.database.windows.net;PORT=1433;DATABASE=KubotaTmapSqlServer;UID=kubotatmapsqladmin;PWD=Tavant123TMAP$"
export SQLAZURECONNSTR_KSQLALCHEMYCONSTR="DRIVER={ODBC Driver 17 for SQL Server};SERVER=kubota-tmap-sql-server.database.windows.net,1433;DATABASE=KubotaTmapSqlServer;UID=kubotatmapsqladmin;PWD=Tavant123TMAP$"
export PATH=${PATH}:${MFTA_ORACLE_CON_STR}:${SQLAZURECONNSTR_KPYODBCCONSTR}:${SQLAZURECONNSTR_KSQLALCHEMYCONSTR}
-----------------------------------------------------------------

source ~/.bashrc
source ~/.bash_profile
source /etc/environment
azurite

https://www.kaggle.com/maunish/osic-super-cool-eda-and-pytorch-baseline

==========================================================


https://www.kaggle.com/sidba20/make-a-successful-android-app
Viz are great

data.loc[10472]=data.loc[10472].shift() # hole shift
#swap fisrt and second column
data['App'].loc[10472] = data['Category'].loc[10472]
data['Category'].loc[10472] = np.nan
data.loc[10472]

data[data.duplicated()].count()

data['Size'].replace('Varies with device', np.nan, inplace = True ) 
data['Size']=data['Size'].str.extract(r'([\d\.]+)', expand=False).astype(float) * \
    data['Size'].str.extract(r'([kM]+)', expand=False).fillna(1).replace(['k','M'],[1,1000]).astype(int)
	
	
data['Installs']=data['Installs'].str.replace(r'\D','').astype(float)

==========================================================

Test for distribution

https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3

https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/#:~:text=A%20simple%20and%20commonly%20used,in%20each%20bin%20is%20retained.

https://mode.com/python-tutorial/python-histograms-boxplots-and-distributions/
