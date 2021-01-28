# %%
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    auc,
    roc_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# %%
sns.scatterplot(x="ratio", y="time", hue="label", data=dataset)
# %%
dataset = pd.read_csv('ecommerce_consumers.csv')  # Other type of file could be used which contains tabular data
dataset['label'] = dataset['label'].apply(lambda x: 1 if x.lower() == 'male' else 0)
# %%
# Target column must be last to work below all cell's code correctly, If you don't have your target colum last then make necessary changes to below two lines of code
X = dataset[['ratio', 'time']]
y = dataset[['label']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=997, test_size=0.2)
# %%
# Do required transformation(s) for X and/or y (If required)
sc_x = StandardScaler()
X_train['time'] = sc_x.fit_transform(X_train['time'].values.reshape(-1, 1))
X_test['time'] = sc_x.transform(X_test['time'].values.reshape(-1, 1))

# %%
classifier = DecisionTreeClassifier(random_state=997).fit(X_train, y_train)
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
