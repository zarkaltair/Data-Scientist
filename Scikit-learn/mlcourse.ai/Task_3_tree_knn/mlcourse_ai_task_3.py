#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data = pd.read_csv('telecom_churn.csv')


# In[4]:


data.head(3)


# In[3]:


data.drop(['State', 'Voice mail plan'], axis=1, inplace=True)


# In[5]:


data['International plan'] = data['International plan'].map({'No': 0, 'Yes': 1})


# In[6]:


data.info()


# In[7]:


y = data['Churn'].astype('int')


# In[8]:


x = data.drop(['Churn'], axis=1)


# In[10]:


x.shape, y.shape


# In[14]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)


# In[19]:


x_train.shape, x_test.shape


# In[20]:


first_tree = DecisionTreeClassifier(random_state=17)


# In[23]:


np.mean(cross_val_score(first_tree, x_train, y_train, cv=5))


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


first_knn = KNeighborsClassifier()


# In[26]:


np.mean(cross_val_score(first_knn, x_train, y_train, cv=5))


# In[27]:


from sklearn.model_selection import GridSearchCV


# In[28]:


tree_params = {'max_depth': np.arange(1, 11), 'max_features': [0.5, 0.7, 1.0]}


# In[29]:


tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)


# In[34]:


get_ipython().run_cell_magic('time', '', 'tree_grid.fit(x_train, y_train)')


# In[35]:


tree_grid.best_params_


# In[36]:


tree_grid.best_score_


# In[50]:


knn_params = {'n_neighbors': list(range(9, 30)) + list(range(50, 100, 10))}


# In[51]:


knn_grid = GridSearchCV(first_knn, knn_params, cv=5, n_jobs=-1)


# In[52]:


get_ipython().run_cell_magic('time', '', 'knn_grid.fit(x_train, y_train)')


# In[53]:


knn_grid.best_params_


# In[56]:


knn_grid.best_score_


# In[57]:


tree_grid.best_estimator_


# In[59]:


tree_preds = tree_grid.predict(x_test)


# In[60]:


from sklearn.metrics import accuracy_score


# In[61]:


accuracy_score(tree_preds, y_test)


# In[62]:


1 - np.mean(y)


# In[63]:


from sklearn.tree import export_graphviz


# In[64]:


export_graphviz(tree_grid.best_estimator_, out_file='telecom_tree.dot', feature_names=x.columns, filled=True)


# In[65]:


get_ipython().system('ls')


# In[66]:


get_ipython().system('dot -Tpng telecom_tree.dot -o telecom_tree.png')

