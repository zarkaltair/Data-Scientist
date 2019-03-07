#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# In[3]:


df = pd.read_csv('gbm-data.csv')
y = df['Activity'].values
X = df.drop(['Activity'], axis=1).values


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


# In[17]:


gbm_model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.2, verbose=True, random_state=241)
gbm_model.fit(X_train, y_train)


# In[18]:


arr = []
for i in gbm_model.staged_decision_function(X_test):
    arr.append(log_loss(y_test, [(1.0 / (1.0 + math.exp(-j))) for j in i]))
min(arr)


# In[27]:


rf_model = RandomForestClassifier(n_estimators=36, random_state=241)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict_proba(X_test)[:, 1]
log_loss(y_test, y_pred)

