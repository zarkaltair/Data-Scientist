#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['figure.figsize'] = 12, 6

data_demo = pd.read_csv('weights_heights.csv')

plt.scatter(data_demo['Weight'], data_demo['Height']);
plt.xlabel('Вес в фунтах')
plt.ylabel('Рост в дюймах');


# In[5]:


import warnings
warnings.filterwarnings('ignore')
import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import fetch_20newsgroups, load_files

import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv('bank_train.csv')
labels = pd.read_csv('bank_train_target.csv', header=None)

df.head()


# In[7]:


df['education'].value_counts().plot.barh(figsize=(12, 6));


# In[9]:


label_encoder = LabelEncoder()

mapped_education = pd.Series(label_encoder.fit_transform(df['education']))
mapped_education.value_counts().plot.barh(figsize=(12, 6))
print(dict(enumerate(label_encoder.classes_)))


# In[10]:


categorical_columns = df.columns[df.dtypes == 'object'].union(['education'])
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
df.head()


# In[11]:


def logistic_regression_accuracy_on(dataframe, labels):
    features = dataframe.as_matrix()
    train_features, test_features, train_labels, test_labels =         train_test_split(features, labels)

    logit = LogisticRegression()
    logit.fit(train_features, train_labels)
    return classification_report(test_labels, logit.predict(test_features))

print(logistic_regression_accuracy_on(df[categorical_columns], labels))


# In[12]:


onehot_encoder = OneHotEncoder(sparse=False)

encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_columns]))
encoded_categorical_columns.head()


# In[13]:


print(logistic_regression_accuracy_on(encoded_categorical_columns, labels))


# In[14]:


for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s))


# In[15]:


hash_space = 25
for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s) % hash_space)


# In[16]:


hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_space)}])
for s in ('job=student', 'marital=single', 'day_of_week=mon'):
    print(s, '->', hash(s) % hash_space)
    hashing_example.loc[0, hash(s) % hash_space] = 1
hashing_example


# In[17]:


assert hash('no') == hash('no')
assert hash('housing=no') != hash('loan=no')

