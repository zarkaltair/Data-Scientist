#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression


# In[11]:


pd.set_option('max_rows', None)
df = pd.read_csv('features.csv', index_col='match_id')
# df.count()


# In[12]:


df_test = pd.read_csv('features_test.csv', index_col='match_id')
df_test.shape


# In[13]:


# passes have the following signs:
passes = ['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 
         'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time', 'dire_bottle_time', 
         'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']
# 'first_blood_time' - time first kill
# 'first_blood_team' - team which make first kill


# In[14]:


df[passes] = df[passes].fillna(0)
df_test[passes] = df_test[passes].fillna(0)


# In[15]:


# target variable
y = df['radiant_win']
X = df.drop(['radiant_win', 'duration', 'tower_status_radiant', 'tower_status_dire', 
             'barracks_status_radiant', 'barracks_status_dire'], axis=1)
X.shape


# In[16]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# In[23]:


get_ipython().run_cell_magic('time', '', 'gbm_model = GradientBoostingClassifier(n_estimators=40, random_state=42)\ngbm_model.fit(X, y)\nscore = cross_val_score(gbm_model, X, y, cv=kfold)')


# In[24]:


score.mean()

