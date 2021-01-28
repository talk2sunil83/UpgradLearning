# %%
from pandas_profiling import ProfileReport
from typing import Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%

TP = 50
FP = 40
TN = 80
FN = 30

# %%
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# %%
# Let us calculate specificity
TN / float(TN+FP)
# %%
# Calculate false positive rate - predicting churn when customer does not have churned
print(FP / float(TN+FP))
# %%
# positive predictive value
print(TP / float(TP+FP))

# %%
# Negative predictive value
print(TN / float(TN + FN))
# %%
TP = 400
FP = 200
TN = 300
FN = 100
# %%
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# %%
# Let us calculate specificity
TN / float(TN+FP)
# %%
# Calculate false positive rate - predicting churn when customer does not have churned
print(FP / float(TN+FP))
# %%
# positive predictive value
print(TP / float(TP+FP))

# %%
# Negative predictive value
print(TN / float(TN + FN))
# %%


def TPR(TP: float, FN: float) -> float:

    return TP / float(TP+FN), f"{TP}/{TP+FN}"


def Sensitivity(TP: float, FN: float) -> float:
    return TPR(TP, FN)


def Specificity(TN: float, FP: float) -> float:
    return TN / float(TN+FP), f"{TN}/{TN+FP}"


def FPR(TN: float, FP: float) -> float:
    return 1 - Specificity(TN, FP)[0]


def Precision(TP: float, FP: float) -> float:
    return TP / float(TP+FP), f"{TP}/{TP+FP}"


def Recall(TP: float, FN: float) -> float:
    return TP / float(TP+FN), f"{TP}/{TP+FN}"


def F1Score(precision: float, recall: float) -> float:
    return (2 * precision*recall) / (precision+recall)


def Accuracy(TN: float, FP: float, FN: float, TP: float) -> float:
    return (TN+TP)/(TN+TP+FP+FN), f"{TN+TP}/{TN+TP+FP+FN}"


def get_classification_matrix(TN: float, FP: float, FN: float, TP: float) -> Dict[str, float]:
    precision = Precision(TP, FP)[0]
    recall = Recall(TP, FN)[0]
    return {
        "TPR": TPR(TP, FN),
        "Sensitivity": Sensitivity(TP, FN),
        "Specificity": Specificity(TN, FP),
        "FPR": FPR(TN, FP),
        "Precision": precision,
        "Recall": recall,
        "F1Score": F1Score(precision, recall),
        "Accuracy": Accuracy(TN, FP, FN, TP)
    }


# %%
# %%
%matplotlib inline
# %%
cutoff_df = pd.read_csv("tradeoff.csv")
# %%

cutoff_df.plot.line(x='Probability', y=[
                    'Accuracy', 'Sensitivity', 'Specificity'])
plt.show()

# %%
TP = 150
FP = 100
TN = 400
FN = 50


for k, v in get_classification_matrix(TN, FP, FN, TP).items():
    print(f"{k}: {v}")

# %%
# %%
%matplotlib inline
# %%
df = pd.read_csv("churn_data.csv")
# %%
profile = ProfileReport(df, title="Pandas Profiling Report")
profile

# %%
df.dropna(inplace=True)

# %%
df["Churn"] = df["Churn"].astype('category')
df["TotalCharges"] = df["TotalCharges"].astype(float)
# %%
sns.boxplot(data=df, x="Churn", y="TotalCharges")
# %%
# %%
TP = 359
FP = 234
TN = 1294
FN = 223


for k, v in get_classification_matrix(TN, FP, FN, TP).items():
    print(f"{k}: {v}")
# %%
# %%
for k, v in get_classification_matrix(1200, 400, 350, 1050).items():
    print(f"{k}: {v}")
# %%
for k, v in get_classification_matrix(1200, 400, 350, 1190).items():
    print(f"{k}: {v}")
# %%
0.7083333333333334 - 0.75
# %%
TN = 3
FP = 2
FN = 1
TP = 4
for k, v in get_classification_matrix(TN, FP, FN, TP).items():
    print(f"{k}: {v}")
# %%
TN = 500
FP = 20
FN = 40
TP = 440
for k, v in get_classification_matrix(TN, FP, FN, TP).items():
    print(f"{k}: {v}")
# %%
