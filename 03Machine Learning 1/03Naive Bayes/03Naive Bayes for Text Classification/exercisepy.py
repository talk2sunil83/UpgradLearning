#!/usr/bin/env python
# coding: utf-8

# ## SMS Spam Classifier: Multinomial Naive Bayes
#
# The notebook is divided into the following sections:
# 1. Importing and preprocessing data
# 2. Building the model: Multinomial Naive Bayes
#     - Model building
#     - Model evaluation

# ### 1. Importing and Preprocessing Data

# In[1]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
# %%

# reading the training data
docs = pd.read_csv('Exercise.csv')
docs.head()


# In[111]:


# number of SMSes / documents
len(docs)


# In[112]:


# counting spam and ham instances
hot_cold = docs.Class.value_counts()
hot_cold


# In[113]:


print("cold rate is about {0}%".format(
    round((hot_cold[1]/float(hot_cold[0]+hot_cold[1]))*100), 2))


# In[114]:


# mapping labels to 0 and 1
docs['label'] = docs.Class.map({'Hot': 0, 'Cold': 1})


# In[115]:


docs.head()


# In[116]:


# we can now drop the column 'Class'
docs = docs.drop('Class', axis=1)
docs.head()


# In[117]:


# convert to X and y
X = docs.Document
y = docs.label
print(X.shape)
print(y.shape)


# In[118]:


# splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[119]:


X_train.head()


# In[120]:


y_train.head()


# In[121]:


# vectorizing the sentences; removing stop words
vect = CountVectorizer(stop_words='english')


# In[122]:


vect.fit(X)
# vect.fit(X_train)


# In[123]:


# printing the vocabulary
vect.vocabulary_


# In[124]:


# vocab size
len(vect.vocabulary_.keys())


# In[125]:


# transforming the train and test datasets
X_train_transformed = vect.transform(X_train)
X_test_transformed = vect.transform(X_test)


# In[126]:


# note that the type is transformed (sparse) matrix
print(type(X_train_transformed))
print(X_train_transformed)


# ### 2. Building and Evaluating the Model

# In[127]:


# training the NB model and making predictions
mnb = MultinomialNB()

# fit
mnb.fit(X_train_transformed, y_train)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba = mnb.predict_proba(X_test_transformed)


# In[143]:


# note that alpha=1 is used by default for smoothing
mnb


# ### Model Evaluation

# In[129]:


# printing the overall accuracy
metrics.accuracy_score(y_test, y_pred_class)


# In[145]:


# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
# help(metrics.confusion_matrix)


# In[131]:


confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[132]:


sensitivity = TP / float(FN + TP)
print("sensitivity", sensitivity)


# In[133]:


specificity = TN / float(TN + FP)
print("specificity", specificity)


# In[134]:


precision = TP / float(TP + FP)
print("precision", precision)
print(metrics.precision_score(y_test, y_pred_class))


# In[135]:


print("precision", precision)
print("PRECISION SCORE :", metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :", metrics.f1_score(y_test, y_pred_class))


# In[136]:


y_pred_class


# In[137]:


y_pred_proba


# In[138]:


# creating an ROC curve

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[139]:


# area under the curve
print(roc_auc)


# In[140]:


# matrix of thresholds, tpr, fpr
pd.DataFrame({'Threshold': thresholds,
              'TPR': true_positive_rate,
              'FPR': false_positive_rate
              })


# In[141]:


# plotting the ROC curve
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)
