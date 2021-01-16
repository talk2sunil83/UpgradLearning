# %% [markdown]
'''
Support Vector Machine (SVM) Classification
'''
# %%
import sklearn
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
# %%
dataset = pd.read_csv('ecommerce_consumers.csv')
dataset['label'] = dataset['label'].apply(lambda x: 1 if x.lower() == 'male' else 0)

# %%
X = dataset[['ratio', 'time']]
y = dataset[['label']]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=997)
# %%
ss = StandardScaler()
X_train['time'] = ss.fit_transform(X_train['time'].values.reshape(-1, 1))
X_test['time'] = ss.transform(X_test['time'].values.reshape(-1, 1))
# Do All required transformations as they are data specific
# %%
print(f"sklearn version: {sklearn.__version__}")
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    classifier = SVC(kernel=kernel, random_state=997).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("-"*25, "> ", kernel, " <", "-"*25)
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
    print(f'Recall Score: {recall_score(y_test, y_pred)}')
    print(f'Classification Report:\n {classification_report(y_test, y_pred)}')
# %%
