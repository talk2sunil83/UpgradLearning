# %%
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path
import os
import sys

%matplotlib inline
# %%

# %%
ROOT_FOLDER: Path = None
if '__file__' in sys.argv:
    ROOT_FOLDER = Path(__file__).resolve().parents[0]  # works in file mode
else:
    ROOT_FOLDER = Path(os.getcwd())  # Works in iteractive mode

ROOT_FOLDER
# %%

xlsx_files = [x for x in Path(ROOT_FOLDER / "data").rglob('*') if x.is_file()]
xlsx_files

# %%
file_names = [f.parts[-1][:-5] for f in xlsx_files]
file_names

# %%
data_map = {df: pd.read_excel(fn) for fn, df in zip(xlsx_files, file_names)}
# %%
for file, frame in data_map.items():
    print(file, frame.columns)

# %%
CLAIM_NUMBER = 'CLAIM_NUMBER'
CLAIM_ID = 'CLAIM_ID'
SERIAL_NUMBER = 'SERIAL_NUMBER'

# %%
claims: DataFrame = data_map['Claims']
claim_amount: DataFrame = data_map['Claim_Amount']
failure_info: DataFrame = data_map['Failure_Info']
job: DataFrame = data_map['Job']
parts_replaced: DataFrame = data_map['Parts_Replaced']
inventory: DataFrame = data_map['Inventory']

claims_with_amount = claims.merge(claim_amount, on=CLAIM_ID, how='inner').merge(
    failure_info, on=CLAIM_NUMBER, how='inner').merge(
    inventory, on=SERIAL_NUMBER, how='inner').merge(
    job, on=CLAIM_NUMBER, how='inner').merge(parts_replaced, on=CLAIM_NUMBER, how='inner')


# %%
claims_with_amount.columns
# %%
claims_with_amount.drop(['CLAIM_ID_y', 'CLAIM_ID_x', ], axis=1, inplace=True)
claims_with_amount.rename({'OTHER_AMT(SERVICES)': 'OTHER_AMT', 'FAULT_DESC (DAMAGE_CODE)': 'FAULT_DESC', 'FAULT_CODE (Position_Code)': 'FAULT_CODE'}, axis=1, inplace=True)
claims_with_amount.columns
# %%
claims_with_amount.info()
# %%
claims_with_amount.head()
# %%
claims_with_amount.isnull().sum()
# %%
claims_with_amount = claims_with_amount[claims_with_amount['HOURS_ON_MACHINE'].notna()]
claims_with_amount = claims_with_amount[claims_with_amount['WNTY_START_DATE'].notna()]
claims_with_amount = claims_with_amount[claims_with_amount['WNTY_END_DATE'].notna()]
claims_with_amount.isnull().sum()

# %%
claims_with_amount['APPLICABLE_POLICY'].value_counts()
# %%
claims_with_amount['APPLICABLE_POLICY'].fillna("NA", inplace=True)
claims_with_amount['APPLICABLE_POLICY'].value_counts()
# %%
claims_with_amount.isnull().sum()
# %%
claims_with_amount['CAUSAL_PART'].value_counts().count()
# %%
claims_with_amount['CAUSAL_PART'].fillna("NA", inplace=True)
claims_with_amount['CAUSAL_PART'].value_counts()
# %%
claims_with_amount.isnull().sum()
# %%
sum(claims_with_amount['ADD_LABOR_HRS'] == 0)

# %%
claims_with_amount['ADD_LABOR_HRS'].fillna(0, inplace=True)
# %%
claims_with_amount.isnull().sum()

# %%
claims_with_amount.info()

# %%
claims_with_amount.drop(['SERIAL_NUMBER', 'DEALER_NUMBER', 'DEALER_NAME', 'PARENT_DEALER', 'FAULT_DESC', 'BUSINESS_UNIT', 'PRODUCT_CODE'], axis=1, inplace=True)
claims_with_amount.info()

# %%
cat_cols = ['CLAIM_TYPE',
            'CLAIM_STATUS',
            'STOCK_RETAIL',
            'APPLICABLE_POLICY',
            'DEALER_CITY',
            'DEALER_STATE',
            'DEALER_COUNTRY',
            'CAUSAL_PART',
            'FAULT_CODE',
            'FAULT_LOCATION',
            'PRODUCT_FAMILY',
            'PRODUCT_NAME',
            'MODEL_CODE',
            'MODEL_NAME',
            'TIME_TYPE_CODE',
            'VARIANT_CODE',
            'VARIANT',
            'JOB_CODE',
            'PART_TYPE',
            'OEM_PART_NUMBER']
# %%
for col in cat_cols:
    print(col)
    claims_with_amount[col] = claims_with_amount[col].str.upper()
# %%
claims_with_amount[cat_cols] = claims_with_amount[cat_cols].astype('category')
# %%
date_cols = [col for col in claims_with_amount.columns if col.lower().endswith('_date')]
claims_with_amount.drop(date_cols, axis=1, inplace=True)
# %%
num_cols = claims_with_amount.select_dtypes(include=['float64', 'int64']).columns
num_cols = [cn for cn in num_cols if cn != CLAIM_NUMBER]

# %%
claims_with_amount.head()
# %%
claims_with_amount_scaled = claims_with_amount.copy()
# %%
mms_scalers = {}
for col in num_cols:
    mms = MinMaxScaler()
    claims_with_amount_scaled[col] = mms.fit_transform(claims_with_amount_scaled[col].values.reshape(-1, 1))
    mms_scalers.update({col: mms})

# %%
claims_with_amount_scaled.head()
# %%
# num_column_trans = ColumnTransformer([(col+'_mms', MinMaxScaler(), [col]) for col in num_cols], remainder='passthrough')
# claims_with_amount_scaled =pd.DataFrame(num_column_trans.fit_transform(claims_with_amount), columns=claims_with_amount.columns)
# %%

# %%
claims_with_amount_scaled.shape
# %%
dummy1 = pd.get_dummies(claims_with_amount_scaled[cat_cols], drop_first=True)
dummy1.shape
# %%
claims_data = pd.concat([claims_with_amount_scaled, dummy1], axis=1)
# %%
claims_data.drop(cat_cols, axis=1, inplace=True)
claims_data.drop(CLAIM_NUMBER, axis=1, inplace=True)
# %%
claims_data.shape

# %%
pickle.dump(claims_data, open("scaled_data.pkl", 'wb'))

# %%
