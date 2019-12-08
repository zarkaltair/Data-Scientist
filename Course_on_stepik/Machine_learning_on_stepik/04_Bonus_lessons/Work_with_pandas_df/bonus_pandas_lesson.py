#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movie = pd.read_csv('movie_metadata.csv')
movie.head()


# In[3]:


genres = movie[['movie_title', 'genres']]


# In[4]:


for i in genres:
    print(i)


# In[5]:


for i in genres.columns:
    print(i)


# In[9]:


def reversator(value):
    return value[:: -1]


# In[10]:


for row in genres.values:
    for value in row:
        print(reversator(value))


# In[13]:


for i, row in genres.iterrows():
    print(row.map(reversator))


# In[18]:


for i, col in genres.iteritems():
    print(col.map(reversator))


# In[20]:


budget = movie[['budget', 'duration']]
budget.head()


# In[21]:


budget.applymap(lambda x: x + 1)


# In[22]:


budget.apply(np.mean, axis=0)


# In[23]:


def mm(col):
    return np.mean(col) + 1


# In[24]:


budget.apply(mm)


# In[25]:


budget.mean() + 1


# In[26]:


budget.max()


# In[29]:


np.mean(budget['budget'].dropna().values)


# In[30]:


get_ipython().run_cell_magic('timeit', '', "budget['budget'].apply(np.mean)")


# In[32]:


get_ipython().run_cell_magic('timeit', '', "budget['budget'].apply('mean')")


# In[34]:


get_ipython().run_cell_magic('timeit', '', "budget['budget'].mean(axis=0)")


# In[35]:


get_ipython().run_cell_magic('timeit', '', "budget['budget'].describe().loc['mean']")

