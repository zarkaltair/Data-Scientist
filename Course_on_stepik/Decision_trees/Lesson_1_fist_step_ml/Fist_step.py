#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import tree
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
sklearn.__version__


# In[2]:


data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1], 'X_2': [0, 0, 0, 1, 0, 0, 0, 1], 'Y': [1, 1, 1, 1, 0, 0, 0, 0]})


# In[3]:


data.head()


# In[4]:


clf = tree.DecisionTreeClassifier(criterion='entropy')


# In[5]:


clf


# In[6]:


X = data[['X_1', 'X_2']]
y = data.Y


# In[7]:


clf.fit(X, y)


# In[8]:


tree.plot_tree(clf, feature_names=list(X),
               class_names=['Negative', 'Positive'],
               filled=True);


# In[29]:


df = pd.read_csv('cats.csv', index_col=0)
df


# In[50]:


X = df[['Гавкает']]
y = df['Вид']


# In[51]:


clf_df = tree.DecisionTreeClassifier(criterion='entropy')


# In[52]:


clf_df.fit(X, y)


# In[53]:


tree.plot_tree(clf_df, feature_names=list(X),
               class_names=['Negative', 'Positive'],
               filled=True);


# In[27]:


- 4 / 9 * np.log2(4 / 9) - 5 / 9 * np.log2(5 / 9)


# In[28]:


- 4 / 5 * np.log2(4 / 5) - 1 / 5 * np.log2(1 / 5)


# In[58]:


0.97 - (0 + 9 / 10 * 0.99)


# In[57]:


0.97 - (0 + 5 / 10 * 0.72)

