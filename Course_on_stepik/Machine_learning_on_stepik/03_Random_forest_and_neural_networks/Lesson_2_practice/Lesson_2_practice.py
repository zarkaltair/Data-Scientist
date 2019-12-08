#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[3]:


df = pd.read_csv('training_mush.csv')
df.head()


# In[16]:


y = df['class']


# In[17]:


X = df.drop('class', axis=1)


# In[4]:


clf_rf = RandomForestClassifier(random_state=0)


# In[6]:


parametrs = {'n_estimators': range(10, 60, 10), 
             'max_depth': range(1, 13, 2), 
             'min_samples_leaf': range(1, 8), 
             'min_samples_split': range(2, 10, 2)}


# In[7]:


search = GridSearchCV(clf_rf, parametrs, cv=3, n_jobs=-1)


# In[18]:


search.fit(X, y)


# In[22]:


search.best_params_


# In[23]:


best_clf_rf = search.best_estimator_


# In[24]:


best_clf_rf.fit(X, y)


# In[27]:


feature_importances_df = pd.DataFrame({'features': list(X),
                                       'feature_importances': best_clf_rf.feature_importances_})


# In[29]:


feature_importances_df.sort_values('feature_importances', ascending=False)


# In[30]:


df_test = pd.read_csv('testing_mush.csv')
df_test.head()


# In[31]:


y_pred = best_clf_rf.predict(df_test)


# In[52]:


df_t = pd.DataFrame(y_pred)


# In[57]:


df_t[0].value_counts()


# In[59]:


df_true = pd.read_csv('testing_y_mush.csv')
df_true.head()


# In[60]:


from sklearn.metrics import confusion_matrix


# In[61]:


confusion_matrix(df_true, y_pred)


# In[64]:


sns.heatmap(confusion_matrix(df_true, y_pred), annot=True,annot_kws={"size": 16});


# In[65]:


df_invasion = pd.read_csv('invasion.csv')
df_invasion.head()


# In[66]:


df_operative_info = pd.read_csv('operative_information.csv')
df_operative_info.head()


# In[68]:


df_invasion['class'].unique()


# In[70]:


df_invasion = df_invasion.replace({'transport': 0, 'fighter': 1, 'cruiser': 2})


# In[71]:


df_invasion.head()


# In[72]:


y = df_invasion['class']
X_train = df_invasion.drop('class', axis=1)


# In[73]:


X_test = df_operative_info


# In[74]:


clf_rf = RandomForestClassifier()


# In[76]:


search = GridSearchCV(clf_rf, parametrs, cv=3, n_jobs=-1)


# In[77]:


search.fit(X_train, y)


# In[78]:


search.best_params_


# In[79]:


best_clf_rf = search.best_estimator_


# In[80]:


best_clf_rf.fit(X_train, y)


# In[81]:


y_pred = best_clf_rf.predict(X_test)


# In[84]:


pd.Series(y_pred).value_counts()


# In[85]:


feature_importances_df = pd.DataFrame({'features': list(X_train),
                                       'feature_importances': best_clf_rf.feature_importances_})


# In[86]:


feature_importances_df.sort_values('feature_importances', ascending=False)


# In[88]:


df_space = pd.read_csv('space_can_be_a_dangerous_place.csv')
df_space.head()


# In[89]:


df_space.shape


# In[90]:


y = df_space['dangerous']
X = df_space.drop('dangerous', axis=1)


# In[91]:


from sklearn.model_selection import RandomizedSearchCV


# In[94]:


clf_rf = RandomForestClassifier()


# In[95]:


rand_search = RandomizedSearchCV(clf_rf, parametrs, cv=3, n_jobs=-1)


# In[98]:


get_ipython().run_line_magic('time', '')
rand_search.fit(X, y)


# In[102]:


best_rand_search = rand_search.best_estimator_


# In[103]:


best_rand_search.fit(X, y)


# In[104]:


feature_importances_df = pd.DataFrame({'features': list(X),
                                       'feature_importances': best_rand_search.feature_importances_})


# In[105]:


feature_importances_df.sort_values('feature_importances', ascending=False)

