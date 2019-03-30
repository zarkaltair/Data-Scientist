#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tqdm
from sklearn.datasets import load_files


# In[2]:


PATH_TO_DATA = 'imdb_reviews'


# In[3]:


get_ipython().system('du -hs $PATH_TO_DATA')


# In[4]:


get_ipython().system('du -hs $PATH_TO_DATA/train')
get_ipython().system('du -hs $PATH_TO_DATA/test')


# In[5]:


get_ipython().run_cell_magic('time', '', "train_reviews = load_files(os.path.join(PATH_TO_DATA, 'train'))")


# In[6]:


get_ipython().run_cell_magic('time', '', "test_reviews = load_files(os.path.join(PATH_TO_DATA, 'test'))")


# In[7]:


type(train_reviews), len(train_reviews.data)


# In[8]:


train_reviews.data[0]


# In[9]:


train_reviews.target[0]


# In[10]:


train_reviews.data[1]


# In[11]:


train_reviews.target[1]


# BOW - Bag of words

# In[12]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# In[13]:


a = np.zeros([5, 5])
a[0, 3] = 1
a[4, 4] = 6
a[2, 2] = 5
a[3, 1] = 4
a[3, 2] = 2
a[1, 1] = 7
a


# In[14]:


pd.DataFrame(a, columns=['apple', 'wax', 'sadness', 'luck', 'girl'])


# In[15]:


b = csr_matrix(a)
b


# In[16]:


b.todense()


# In[17]:


b.nonzero()


# In[18]:


b.data


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[45]:


get_ipython().run_cell_magic('time', '', 'cv = CountVectorizer(ngram_range=(1, 2))\nX_train_sparse = cv.fit_transform(train_reviews.data)')


# In[46]:


len(cv.vocabulary_)


# In[47]:


get_ipython().run_cell_magic('time', '', 'X_test_sparse = cv.transform(test_reviews.data)')


# In[48]:


X_train_sparse.shape, X_test_sparse.shape


# In[24]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score


# In[25]:


y_train, y_test = train_reviews.target, test_reviews.target


# In[26]:


np.bincount(y_train), np.bincount(y_test)


# In[29]:


10 ** 6 / X_train_sparse.shape[0]


# In[40]:


logit = LogisticRegression(random_state=17, n_jobs=2, solver='lbfgs')
sgd_logit = SGDClassifier(max_iter=40, random_state=17, n_jobs=2, tol=1e-3)


# In[51]:


get_ipython().run_cell_magic('time', '', 'logit.fit(X_train_sparse, y_train)')


# In[49]:


get_ipython().run_cell_magic('time', '', 'sgd_logit.fit(X_train_sparse, y_train)')


# In[52]:


accuracy_score(y_test, logit.predict(X_test_sparse))


# In[50]:


accuracy_score(y_test, sgd_logit.predict(X_test_sparse))

