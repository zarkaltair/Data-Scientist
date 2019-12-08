#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


students_performance = pd.read_csv('StudentsPerformance.csv')


# In[6]:


students_performance.head(3)


# In[8]:


students_performance.describe()


# In[10]:


students_performance.dtypes


# In[11]:


students_performance.shape


# In[12]:


students_performance.size


# In[14]:


students_performance.iloc[0:3, 0:5]


# In[16]:


students_performance.loc[:6]


# In[17]:


students_performance.head(7)


# In[18]:


students_performance.iloc[0:7]


# In[19]:


titanic_df = pd.read_csv('titanic.csv')


# In[20]:


titanic_df.shape


# In[21]:


titanic_df.dtypes


# In[22]:


titanic_df.info()


# In[24]:


students_performance_with_names = students_performance.iloc[[0, 3, 4, 7, 8]]


# In[25]:


students_performance_with_names.index = ['Cersei', 'Tywin', 'Gregor', 'Joffrey', 'Ilen Pain']


# In[26]:


students_performance_with_names


# In[27]:


students_performance_with_names.loc[['Cersei', 'Joffrey']]


# In[29]:


students_performance_with_names.loc[['Cersei', 'Joffrey'], ['gender', 'writing score']]


# In[31]:


students_performance_with_names.iloc[:, 0]


# In[32]:


students_performance_with_names['gender']


# In[33]:


students_performance_with_names[['gender']]


# In[ ]:





# In[ ]:




