
# %% [markdown]
'''

Logistic Regression Classifier
Random forest Classifier
Decision Tree Classifier
PCA
    Logistic Regression Classifier
    Random forest Classifier
    Decision Tree Classifier

Normalization
Modeling
    With PCA
    Without PCA
'''
all_metrics.append(["Logistic Regression", round(sensitivity, 2), round(specificity, 2), round(roc_auc_score(y_test, y_pred_prob), 2)])
# %%
total_rech_amts = ['total_rech_amt_6', 'total_rech_amt_7', 'total_rech_amt_8', 'total_rech_amt_9']
av_rech_amts = ['av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9']

percentiles = [0.1,  0.25, 0.5,  0.6, 0.7, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]
# %%
churn_data[av_rech_amts].describe(percentiles=percentiles).T
# %%
churn_data[total_rech_amts].describe(percentiles=percentiles).T

# %%
["mobile_number", "circle_id", "loc_og_t2o_mou", "std_og_t2o_mou", "loc_ic_t2o_mou", "last_date_of_month_6", "arpu_6", "onnet_mou_6", "offnet_mou_6", "roam_ic_mou_6", "roam_og_mou_6", "loc_og_t2t_mou_6", "loc_og_t2m_mou_6", "loc_og_t2f_mou_6", "loc_og_t2c_mou_6", "loc_og_mou_6", "std_og_t2t_mou_6", "std_og_t2m_mou_6", "std_og_t2f_mou_6", "std_og_t2c_mou_6", "std_og_mou_6", "isd_og_mou_6", "spl_og_mou_6", "og_others_6", "total_og_mou_6", "loc_ic_t2t_mou_6", "loc_ic_t2m_mou_6", "loc_ic_t2f_mou_6", "loc_ic_mou_6", "std_ic_t2t_mou_6",
    "std_ic_t2m_mou_6", "std_ic_t2f_mou_6", "std_ic_t2o_mou_6", "std_ic_mou_6", "total_ic_mou_6", "spl_ic_mou_6", "isd_ic_mou_6", "ic_others_6", "total_rech_num_6", "total_rech_amt_6", "max_rech_amt_6", "date_of_last_rech_6", "last_day_rch_amt_6", "date_of_last_rech_data_6", "total_rech_data_6", "max_rech_data_6", "count_rech_2g_6", "count_rech_3g_6", "av_rech_amt_data_6", "vol_2g_mb_6", "vol_3g_mb_6", "arpu_3g_6", "arpu_2g_6", "night_pck_user_6", "monthly_2g_6", "sachet_2g_6", "monthly_3g_6", "sachet_3g_6", "fb_user_6", "aon", "jun_vbc_3g"]

# 'loc_og_t2o_mou':0, 'std_og_t2o_mou':
# %% [markdown]
'''

 - High level data
    1. Call Usage
    2. Money Generate
    3. Recharge Details
    4. Internet Data Usage
    5.
'''
# %% [markdown]
'''

 - Feature extraction
    1. Ratios
    2. Standardized Sum of features, OR Cumulative Sum/Ratio
    3. First or second part of month
    4. Month length influence the total recharge
    5. Revenue based churn- Low Revenue and High incoming calls
    6. (Data usage amount/ Total Usage)*100

*All above analysis will be based on buckets*
'''
# %%
churn_data.shape
# %%
#  churn selection column
ch_cols = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']
churn_data[ch_cols].isnull().sum()
# %%
# total_rech_amt_6 + total_data_rech_6
# total_rech_amt_7 + total_data_rech_7
# AVG,Quantile
#  Impute then filter
# Load ==> impute ==> Filter ==> Feature Extraction


# %%
['arpu_', 'onnet_mou_', 'offnet_mou_', 'roam_ic_mou_', 'roam_og_mou_', 'loc_og_t2t_mou_', 'loc_og_t2m_mou_', 'loc_og_t2f_mou_', 'loc_og_t2c_mou_', 'loc_og_mou_', 'std_og_t2t_mou_', 'std_og_t2m_mou_', 'std_og_t2f_mou_', 'std_og_mou_', 'isd_og_mou_', 'spl_og_mou_', 'og_others_', 'total_og_mou_', 'loc_ic_t2t_mou_', 'loc_ic_t2m_mou_', 'loc_ic_t2f_mou_', 'loc_ic_mou_',
    'std_ic_t2t_mou_', 'std_ic_t2m_mou_', 'std_ic_t2f_mou_', 'std_ic_mou_', 'total_ic_mou_', 'spl_ic_mou_', 'isd_ic_mou_', 'ic_others_', 'total_rech_num_', 'total_rech_amt_', 'max_rech_amt_', 'last_day_rch_amt_', 'total_rech_data_', 'max_rech_data_', 'av_rech_amt_data_', 'vol_2g_mb_', 'vol_3g_mb_', 'monthly_2g_', 'sachet_2g_', 'monthly_3g_', 'sachet_3g_', 'vbc_3g_']

# aon
