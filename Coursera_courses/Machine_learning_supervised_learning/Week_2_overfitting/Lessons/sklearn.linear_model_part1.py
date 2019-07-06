#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## sklearn.liner_model

# **linear_model:**
# * RidgeClassifier
# * SGDClassifier
# * SGDRegressor
# * LinearRegression
# * LogisticRegression
# * Lasso
# * etc

# документация: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# 
# примеры: http://scikit-learn.org/stable/modules/linear_model.html#linear-model

# In[7]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Генерация данных

# In[12]:


blobs = datasets.make_blobs(centers=2, cluster_std=5.5, random_state=1)


# In[13]:


colors = ListedColormap(['red', 'blue'])

plt.figure(figsize=(8, 8))
plt.scatter([x[0] for x in blobs[0]], [x[1] for x in blobs[0]], c=blobs[1], cmap=colors);


# In[16]:


train_data, test_data, train_labels, test_labels = train_test_split(blobs[0], blobs[1], 
                                                                    test_size=0.3,
                                                                    random_state=1)


# ### Линейная классификация

# #### RidgeClassifier

# In[18]:


#создание объекта - классификатора
ridge_classifier = linear_model.RidgeClassifier(random_state=1)


# In[19]:


#обучение классификатора
ridge_classifier.fit(train_data, train_labels)


# In[20]:


#применение обученного классификатора
ridge_predictions = ridge_classifier.predict(test_data)


# In[22]:


print(test_labels)


# In[23]:


print(ridge_predictions)


# In[24]:


#оценка качества классификации
metrics.accuracy_score(test_labels, ridge_predictions)


# In[25]:


ridge_classifier.coef_


# In[26]:


ridge_classifier.intercept_ 


# #### LogisticRegression

# In[30]:


log_regressor = linear_model.LogisticRegression(random_state=1, solver='lbfgs')


# In[31]:


log_regressor.fit(train_data, train_labels)


# In[32]:


lr_predictions = log_regressor.predict(test_data)


# In[33]:


lr_proba_predictions = log_regressor.predict_proba(test_data)


# In[39]:


test_labels


# In[40]:


lr_predictions


# In[41]:


print(lr_proba_predictions)


# In[42]:


metrics.accuracy_score(test_labels, lr_predictions)


# In[43]:


metrics.accuracy_score(test_labels, ridge_predictions)


# ### Оценка качества по cross-validation

# #### cross_val_score

# In[46]:


from sklearn.model_selection import cross_val_score


# In[47]:


ridge_scoring = cross_val_score(ridge_classifier, blobs[0], blobs[1], scoring = 'accuracy', cv = 10)


# In[49]:


lr_scoring = cross_val_score(log_regressor, blobs[0], blobs[1], scoring = 'accuracy', cv = 10)


# In[50]:


lr_scoring


# In[51]:


print('Ridge mean:{}, max:{}, min:{}, std:{}'.format(ridge_scoring.mean(), ridge_scoring.max(), 
                                                     ridge_scoring.min(), ridge_scoring.std()))


# In[52]:


print('Log mean:{}, max:{}, min:{}, std:{}'.format(lr_scoring.mean(), lr_scoring.max(), 
                                                   lr_scoring.min(), lr_scoring.std()))


# #### cross_val_score с заданными scorer и cv_strategy

# In[62]:


scorer = metrics.make_scorer(metrics.accuracy_score)


# In[63]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[73]:


cv_strategy = StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=2)


# In[75]:


ridge_scoring = cross_val_score(ridge_classifier, blobs[0], blobs[1], scoring=scorer, cv=cv_strategy)


# In[76]:


lr_scoring = cross_val_score(log_regressor, blobs[0], blobs[1], scoring=scorer, cv=cv_strategy)


# In[77]:


print('Ridge mean:{}, max:{}, min:{}, std:{}'.format(ridge_scoring.mean(), ridge_scoring.max(), 
                                                     ridge_scoring.min(), ridge_scoring.std()))


# In[78]:


print('Log mean:{}, max:{}, min:{}, std:{}'.format(lr_scoring.mean(), lr_scoring.max(), 
                                                   lr_scoring.min(), lr_scoring.std()))

