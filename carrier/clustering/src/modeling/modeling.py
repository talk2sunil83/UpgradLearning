# %%
import shap
from alibi.explainers import ALE, plot_ale
import alibi
from sklearn.inspection import permutation_importance
from pandas.core.arrays.sparse import dtype
from xgboost import data
import set_base_path
import time
import datetime

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
#  Moldeling Libraries
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
# Scoring Libraries
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score

from IPython.display import display
import xgboost as xgb

import src.utils.modeling as mu

import pandas as pd
import numpy as np
from enum import Enum, auto
import warnings


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from src.constants import *

warnings.filterwarnings('ignore')
%matplotlib inline
# %% [markdown]
'''
# Pandas settings
'''
# %%
pd.options.display.max_columns = None
pd.options.display.max_rows = 500
pd.options.display.width = None
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3

# %% [markdown]
'''
### read data
'''

# %%
merged_df = pd.read_feather(INTERIM_DATA_PATH/"merged_df.feather")
merged_df_scaled = pd.read_feather(INTERIM_DATA_PATH/"merged_df_scaled.feather")
merged_df_dimred = pd.read_feather(INTERIM_DATA_PATH/"merged_df_dimred.feather")


target_col_name = 'IS_SUSPECTED'
# %% [markdown]
'''
# Train test splitResidual Analysis
'''
# %%
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = mu.split_data(merged_df_scaled.drop(target_col_name, axis=1), merged_df_scaled[[target_col_name]])
X_train_dim_red, X_test_dim_red, y_train_dim_red, y_test_dim_red = mu.split_data(merged_df_dimred, merged_df_dimred[[target_col_name]])

# %%
print(
    (y_train_encoded.sum()*100)/len(y_train_encoded),
    (y_test_encoded.sum()*100)/len(y_test_encoded),
    (y_train_dim_red.sum()*100)/len(y_train_dim_red),
    (y_test_dim_red.sum()*100)/len(y_test_dim_red),
)

# %% [markdown]
'''
# Class Balancing
'''
# %%
sm = SMOTE(random_state=42)
X_res_enc, y_res_enc = sm.fit_resample(X_train_encoded, y_train_encoded)
X_res_dim_red, y_res_dim_red = sm.fit_resample(X_train_dim_red, y_train_dim_red)

# %%
print(
    (y_res_enc.sum()*100)/len(y_res_enc),
    (y_res_dim_red.sum()*100)/len(y_res_dim_red)
)


# %% [markdown]
'''
# Create model
'''
# %%
# fit all models
clf_enc = LazyClassifier(predictions=True,  save_model=True, base_path=ENC_MODEL_PATH)
models_res_enc, predictions_enc = clf_enc.fit(X_res_enc, X_test_encoded, y_res_enc, y_test_encoded)
models_enc = clf_enc.provide_models(X_res_enc, X_test_encoded, y_res_enc, y_test_encoded)
# %%
clf_dim_red = LazyClassifier(predictions=True, save_model=True, base_path=DIMRED_MODEL_PATH)
models_res_dim_red, predictions_dim_red = clf_dim_red.fit(X_res_dim_red, X_test_dim_red, y_res_dim_red, y_test_dim_red)
models_dim_red = clf_dim_red.provide_models(X_res_dim_red, X_test_dim_red, y_res_dim_red, y_test_dim_red)

# %%
print(f"Models with encoded data")
display(models_res_enc)
# %%
print(f"Models with dimensionality reduced data")
display(models_res_dim_red)

# %%
linear = 'linear'
rf = 'rf'
model_obj = 'model'
params_dict = 'params'
models = {
    linear: {model_obj: LogisticRegression(random_state=42, n_jobs=-1), params_dict: dict(
        penalty=['l1', 'l2', 'elasticnet', 'none'],
        tol=[1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2],
        C=[1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2],
        solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    )},
    rf: {model_obj: ExtraTreesClassifier(random_state=42, n_jobs=-1), params_dict: dict(
        n_estimators=[10, 20, 50, 100, 200, 500, 1000],
        criterion=["gini", "entropy"],
        max_depth=[2, 3, 5, 10, 20, 50, 100, 200],
        min_samples_split=[2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
        min_samples_leaf=[2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
        max_features=["auto", "sqrt", "log2"],
        max_leaf_nodes=[10, 20, 50, 100, 200, 500, 1000]
    )}
}

# %%


show_graphs = True

trained_models = []

all_model_metrics = []
for model, model_config in models.items():
    print(f"Starting for {model.title()}")
    # With and without Dimensionality Reduction
    for mode in [mu.DataMode.ENCODED, mu.DataMode.ENCODED_DIM_RED]:
        X_train, X_test, y_train, y_test = (X_res_enc, X_test_encoded, y_res_enc, y_test_encoded) if mode == mu.DataMode.ENCODED else (
            X_res_dim_red, X_test_dim_red, y_res_dim_red, y_test_dim_red)
        data_mode_str = "without PCA" if mode == mu.DataMode.ENCODED else "with PCA"
        # Hyper- Parameter tuning
        ra_s_cv = RandomizedSearchCV(model_config.get(model_obj), model_config.get(params_dict), random_state=0, n_jobs=-1, cv=3, verbose=3, return_train_score=True)

        start_time = time.perf_counter()
        ra_s_cv.fit(X_train, y_train)
        end_time = time.perf_counter()

        train_pred = ra_s_cv.predict(X_train)
        test_pred = ra_s_cv.predict(X_test)

        print("-"*50)
        print(f"Best Estimator for {model} {data_mode_str} is {ra_s_cv.best_estimator_}\n")
        print(f"Best Params for {model} {data_mode_str} are {ra_s_cv.best_params_}\n")
        print(f"Cross validation Results for {model} {data_mode_str}\n")
        display(pd.DataFrame(ra_s_cv.cv_results_))
        print("-"*50)

        # Plot evaluation matrix
        mu.prepare_error_data_and_plot(ra_s_cv, X_train, y_train, train_pred, mode, model, show_graphs)
        mu.prepare_error_data_and_plot(ra_s_cv, X_test, y_test, test_pred, mode, model, show_graphs)
        # Collect Models
        model_string = f"{model.title()} {data_mode_str if mode == mu.DataMode.ENCODED_DIM_RED else ''}"
        trained_models.append((model_string, ra_s_cv.best_estimator_, mode))
        # Record performance
        all_model_metrics.append([model_string, ra_s_cv.best_score_, ra_s_cv.score(X_train, y_train),
                                  ra_s_cv.score(X_test, y_test.values),
                                  accuracy_score(y_train, train_pred),
                                  accuracy_score(y_test, test_pred),
                                  balanced_accuracy_score(y_train, train_pred),
                                  balanced_accuracy_score(y_test, test_pred),
                                  roc_auc_score(y_train, train_pred),
                                  roc_auc_score(y_test, test_pred),
                                  f1_score(y_train, train_pred),
                                  f1_score(y_test, test_pred),

                                  #   r2_score(y_train, train_pred), r2_score(y_test, test_pred), # Metrics on train and test
                                  end_time-start_time])
        print("="*50, "\n")
# %%

perf_frame = pd.DataFrame(all_model_metrics, columns=["Algo", "Best Training Score (CV)", "Train Score", "Test Score",
                                                      "Train Accuracy", "Test Accuracy",
                                                      "Balanced Train Accuracy", "Balanced Test Accuracy",
                                                      "Train ROC AUC", "Test ROC AUC",
                                                      "Train F1 Score", "Test F1 Score",
                                                      "Time Taken(Sec)"])

perf_frame.sort_values(by=["Test F1 Score", "Train F1 Score", "Balanced Test Accuracy", "Test Accuracy", "Time Taken(Sec)"], ascending=[False, False, False, False, True]).style.hide_index()


# %%
# for name, trained_model, mode in trained_models:
#     print(name)
#     X_train, X_test, y_train, y_test = (X_res_enc, X_test_encoded, y_res_enc, y_test_encoded) if mode == mu.DataMode.ENCODED else (
#         X_res_dim_red, X_test_dim_red, y_res_dim_red, y_test_dim_red)

#     if hasattr(trained_model, "feature_importances_"):
#         print(trained_model.feature_importances_)
#     if hasattr(trained_model, "coef_"):
#         print(trained_model.coef_)

#     mu.get_feature_importance(trained_model, X_test, y_test)

# %%
best_model = trained_models[2][1]
fe = pd.DataFrame({"Column": X_res_enc.columns, "Importance": best_model.feature_importances_})
fe.sort_values(by="Importance", key=abs, ascending=False).head(20).style.hide_index()


# %%
shap.initjs()
# %%
explainer = shap.TreeExplainer(best_model, data=X_res_enc)
shap_res = explainer.shap_values(X_res_enc, y_res_enc)

# %%
i = 0
shap.force_plot(explainer.expected_value[i], shap_res[i][0], features=X_res_enc.loc[i],
                feature_names=X_res_enc.columns)
# %%
shap.summary_plot(shap_res, X_res_enc, X_res_enc.columns)
# features: Our training set of independent variables
# feature_names: list of column names from the above training set
# %% [markdown]
'''
# Best Model Selection
'''

# %% [markdown]
'''
# Tune the best model
'''

# %% [markdown]
'''
# Conclusion
'''
