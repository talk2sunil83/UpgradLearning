# Copyright 2021 Sunil Yadav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import os
import sqlalchemy
from sqlalchemy import create_engine
import pyodbc
import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


class Db:
    def __init__(self, con_str: str) -> None:
        self.engine = create_engine(con_str, fast_executemany=True)

    def __drop_table__(self, table_name: str) -> None:
        with self.engine.connect() as connection:
            with connection.begin():
                # try:
                # query = f"TRUNCATE TABLE {table_name}"
                # drop table if exists mytablename
                query = f"DROP TABLE IF EXISTS {table_name}"
                connection.execute(query)
                # except Exception as ex:
                #     print(ex)
                #     pass

    def push_data_to_db(self, table_name: str, df: pd.DataFrame, drop_table: bool = True, schema: str = "dbo") -> None:
        if drop_table:
            self.__drop_table__(table_name)
            print(f"Dropped {table_name}")
        df_num_of_cols = len(df.columns)
        chunknum = (2097 // df_num_of_cols) - 1
        df.to_sql(
            table_name,
            con=self.engine,
            index=False,
            schema=schema,
            if_exists="replace",
            method="multi",
            chunksize=chunknum,
        )


con_str = 'mssql+pyodbc://DEVD\SQLEXPRESS/NewTemp?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
db = Db(con_str)

data_path = "E:/Repositories/proptech_model/data/RawData"
# %%

for dirpath, dirnames, filenames in os.walk(data_path):
    for text_file in [file_name for file_name in filenames if file_name.lower().endswith('.txt')]:
        file_path = dirpath + "/" + text_file
        df = pd.read_csv(file_path, sep="\t")
        print(f"Processing {text_file}")
        display(df.dtypes)
        display(df.shape)
        table_name = text_file.replace(" ", "").capitalize().split('.')[0]
        print(table_name)
        db.push_data_to_db(table_name, df)
        print(f"Pushed data for {table_name}")
# %%
