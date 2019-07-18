#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## sklearn.linear_model

# In[27]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Линейная регрессия

# #### Генерация данных

# In[5]:


data, target, coef = datasets.make_regression(n_features=2, n_informative=1, n_targets=1, 
                                              noise=5, coef=True, random_state=2)


# In[16]:


plt.scatter([i[0] for i in data], target, color='r')

plt.scatter([i[1] for i in data], target, color='b');


# In[19]:


train_data, test_data, train_labels, test_labels = train_test_split(data, target,  
                                                                    test_size = 0.3)


# #### LinearRegression

# In[20]:


linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(train_data, train_labels)
predictions = linear_regressor.predict(test_data)


# In[21]:


test_labels


# In[22]:


predictions


# In[23]:


metrics.mean_absolute_error(test_labels, predictions)


# In[30]:


linear_scoring = cross_val_score(linear_regressor, data, target, 
                                 scoring='neg_mean_absolute_error', cv=10)
print('mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std()))


# In[31]:


scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better = True)


# In[33]:


linear_scoring = cross_val_score(linear_regressor, data, target, scoring=scorer, cv = 10)
print('mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std()))


# In[34]:


coef


# In[35]:


linear_regressor.coef_


# In[36]:


# в лекции не указано, что в уравнении обученной модели также участвует свободный член
linear_regressor.intercept_


# In[39]:


print("y = {:.2f} * x1 + {:.2f} * x2".format(coef[0], coef[1]))


# In[40]:


print("y = {:.2f} * x1 + {:.2f} * x2 + {:.2f}".format(linear_regressor.coef_[0], 
                                                  linear_regressor.coef_[1], 
                                                  linear_regressor.intercept_))


# #### Lasso

# In[41]:


lasso_regressor = linear_model.Lasso(random_state=3)
lasso_regressor.fit(train_data, train_labels)
lasso_predictions = lasso_regressor.predict(test_data)


# In[43]:


lasso_scoring = cross_val_score(lasso_regressor, data, target, scoring=scorer, cv=10)
print('mean: {}, std: {}'.format(lasso_scoring.mean(), lasso_scoring.std()))


# In[44]:


lasso_regressor.coef_


# In[45]:


print("y = {:.2f} * x1 + {:.2f} * x2".format(coef[0], coef[1]))


# In[46]:


print("y = {:.2f} * x1 + {:.2f} * x2".format(lasso_regressor.coef_[0], lasso_regressor.coef_[1]))

