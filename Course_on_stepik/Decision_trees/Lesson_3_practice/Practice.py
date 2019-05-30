#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


# In[4]:


df = pd.read_csv('train_data_tree.csv')
df.head()


# In[5]:


clf = DecisionTreeClassifier(criterion='entropy')


# In[6]:


X = df[['sex', 'exang']]
y = df.num


# In[7]:


clf.fit(X, y)


# In[10]:


plot_tree(clf, feature_names=list(X),
          class_names=['Yes', 'No'],
          filled=True);


# In[22]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[23]:


iris = load_iris()
x = iris.data
y = iris.target


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
dt = DecisionTreeClassifier()


# In[33]:


dt.fit(X_train, y_train)
predicted = dt.predict(X_test)


# In[37]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

clf = DecisionTreeClassifier()
parametrs = {'max_depth': range(1, 11), 'min_samples_split': range(2, 11), 'min_samples_leaf': range(1, 11)}
search = GridSearchCV(clf, parametrs, cv=5)
search.fit(x, y)


# In[38]:


search.estimator


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




