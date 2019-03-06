#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import Ridge


# In[2]:


train = pd.read_csv('salary-train.csv')
test = pd.read_csv('salary-test-mini.csv')


# In[3]:


train['FullDescription'] = train['FullDescription'].str.lower()
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

test['FullDescription'] = test['FullDescription'].str.lower()
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)


# In[4]:


vectorizer = TfidfVectorizer(min_df=5)
train_text = vectorizer.fit_transform(train['FullDescription'])

test_text = vectorizer.transform(test['FullDescription'])


# In[5]:


train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)


# In[6]:


dv = DictVectorizer()
train_ohe = dv.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_ohe = dv.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))


# In[7]:


train_x = hstack([train_text, train_ohe])
test_x = hstack([test_text, test_ohe])


# In[10]:


clf = Ridge(alpha=1, random_state=241)
y = train['SalaryNormalized']
clf.fit(train_x, y)
pred = clf.predict(test_x)


# In[11]:


pred

