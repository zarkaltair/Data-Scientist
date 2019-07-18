#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## sklearn.grid_search

# документация: http://scikit-learn.org/stable/modules/grid_search.html

# In[12]:


from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd


# ### Генерация датасета

# In[13]:


iris = datasets.load_iris()


# In[14]:


train_data, test_data, train_labels, test_labels = train_test_split(iris.data, iris.target, 
                                                                    test_size = 0.3,random_state = 0)


# ### Задание модели

# In[15]:


classifier = linear_model.SGDClassifier(random_state=0)


# ### Генерация сетки

# In[16]:


classifier.get_params().keys()


# In[33]:


parameters_grid = {
    'loss' : ['hinge', 'log', 'squared_hinge', 'squared_loss'],
    'penalty' : ['l1', 'l2'],
    'max_iter': range(50, 100),
    'alpha' : np.linspace(0.0001, 0.001, num=5),
}


# In[34]:


cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


# ### Подбор параметров и оценка качества

# #### Grid search

# In[35]:


grid_cv = GridSearchCV(classifier, parameters_grid, scoring='accuracy', cv=cv)


# In[36]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(train_data, train_labels)')


# In[37]:


grid_cv.best_estimator_


# In[38]:


print(grid_cv.best_score_)
print(grid_cv.best_params_)


# In[42]:


grid_cv.cv_results_


# #### Randomized grid search

# In[43]:


from sklearn.model_selection import RandomizedSearchCV


# In[45]:


randomized_grid_cv = RandomizedSearchCV(classifier, parameters_grid, scoring='accuracy',
                                        cv=cv, n_iter=20, random_state=0)


# In[46]:


get_ipython().run_cell_magic('time', '', 'randomized_grid_cv.fit(train_data, train_labels)')


# In[47]:


print(randomized_grid_cv.best_score_)
print(randomized_grid_cv.best_params_)

