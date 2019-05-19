#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


students_performance = pd.read_csv('StudentsPerformance.csv')


# In[6]:


students_performance.head()


# In[10]:


students_performance.loc[students_performance.gender == 'female', ['gender', 'writing score']].head(5)


# In[12]:


mean_writing_score = students_performance['writing score'].mean()


# In[14]:


students_performance.loc[students_performance['writing score'] > mean_writing_score].head()


# In[15]:


query = (students_performance['writing score'] > mean_writing_score) | (students_performance.gender == 'female')


# In[16]:


students_performance.loc[query].head()


# In[20]:


students_performance['lunch'].value_counts() / 1000


# In[32]:


students_performance.loc[students_performance['lunch'] == 'standard', ['math score', 'reading score',
                                                                       'writing score']].mean()


# In[33]:


students_performance.loc[students_performance['lunch'] == 'standard', ['math score', 'reading score',
                                                                       'writing score']].var()


# In[ ]:





# In[36]:


students_performance.loc[students_performance['lunch'] == 'free/reduced', ['math score', 'reading score',
                                                                           'writing score']].mean()


# In[37]:


students_performance.loc[students_performance['lunch'] == 'free/reduced', ['math score', 'reading score',
                                                                           'writing score']].var()


# In[38]:


students_performance.columns = [x.replace(" ", "_") for x in students_performance.columns]


# In[39]:


students_performance.head()


# In[42]:


students_performance.query('writing_score > 74').head()


# In[47]:


writing_score_query = 90


# In[48]:


students_performance.query('writing_score > @writing_score_query').head()


# In[ ]:





# In[49]:


students_performance[['math_score', 'reading_score']].head()


# In[50]:


list(students_performance)


# In[52]:


score_columns = [i for i in list(students_performance) if 'score' in i]


# In[53]:


students_performance[score_columns].head()


# In[55]:


students_performance.filter(like='score').head()


# In[57]:


students_performance.filter(like='_').head()

