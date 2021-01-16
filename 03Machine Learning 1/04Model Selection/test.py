# %%

import sys
import ast
import pandas as pd
from collections import OrderedDict
from datetime import datetime, timedelta
input_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9], 3]

lis = input_list[0]
k = input_list[1]
# %%
input_str = "[2017,1,1,2017,1,1]"
input_list = ast.literal_eval(input_str)
dateStart = datetime.date(input_list[0], input_list[1], input_list[2])
dateEnd = datetime.date(input_list[3], input_list[4], input_list[5])

# OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in xrange((end - start).days)).keys()
# print([dateStart + timedelta(_).strftime(r"%B") for _ in range((dateEnd - dateStart).days)])
# %%
pd.date_range(dateStart, dateEnd, freq='MS').strftime("%B").tolist()
# %%
