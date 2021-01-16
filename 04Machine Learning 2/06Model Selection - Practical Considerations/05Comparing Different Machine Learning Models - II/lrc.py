# %% [markdown]
'''
### Logistic Regression
'''
# %%
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    auc,
    roc_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
dataset = pd.read_csv('ecommerce_consumers.csv')
# %%
dataset.describe()
# %%
dataset.head()
# %%
dataset.info()

# %%
dataset['label'] = dataset['label'].apply(lambda x: 1 if x.lower() == 'male' else 0)
# %%
dataset.head()
# %%
X = dataset[['ratio', 'time']]
y = dataset[['label']]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=997, test_size=0.2)

# %%
# Better scale all features
sc_x = StandardScaler()
X_train['time'] = sc_x.fit_transform(X_train['time'].values.reshape(-1, 1))
X_test['time'] = sc_x.transform(X_test['time'].values.reshape(-1, 1))

# Do All required transformations as they are data specific
# %%
classifier = LogisticRegression(random_state=997).fit(X_train, y_train)
# %%

y_pred = classifier.predict(X_test)
# %%
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc(fpr, tpr)
# %%
print(f'Confusion Matrix:{confusion_matrix(y_test, y_pred)}')
print(f'F1 Score:{f1_score(y_test, y_pred)}')
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'Classification Report :\n {classification_report(y_test, y_pred)}')

# %%
