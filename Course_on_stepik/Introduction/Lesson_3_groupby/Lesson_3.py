#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


students_performance = pd.read_csv('StudentsPerformance.csv')


# In[3]:


students_performance.head()


# In[4]:


students_performance.groupby('gender').mean()


# In[5]:


students_performance.groupby('gender', as_index=False).aggregate(
    {'math score': 'mean', 'reading score': 'mean'}).rename(columns={
    'math score': 'mean math score', 'reading score': 'mean reading score'})


# In[6]:


mean_scores = students_performance.groupby(['gender', 'race/ethnicity']).aggregate(
    {'math score': 'mean', 'reading score': 'mean'}).rename(columns={
    'math score': 'mean math score', 'reading score': 'mean reading score'})


# In[8]:


mean_scores.index


# In[9]:


mean_scores.loc[('female', 'group A')]


# In[10]:


mean_scores.loc[[('female', 'group A'), ('female', 'group B')]]


# In[12]:


students_performance['math score'].unique()


# In[13]:


students_performance['math score'].nunique()


# In[15]:


students_performance.groupby(['gender', 'race/ethnicity'])['math score'].nunique()


# In[19]:


students_performance.sort_values(['gender','math score'], ascending=False).groupby('gender').head(5)


# In[21]:


students_performance['total score'] = students_performance['math score'] + students_performance['reading score'] + students_performance['writing score']


# In[23]:


students_performance.head()


# In[24]:


students_performance = students_performance.assign(total_score_log = np.log(students_performance['total score']))


# In[25]:


students_performance.head()


# In[27]:


students_performance.drop(['total score'], axis=1).head()


# In[ ]:





# In[30]:


df = pd.read_csv('dota_hero_stats.csv', index_col=False)


# In[31]:


df.head()


# In[44]:


df.groupby('legs').size()


# In[45]:


df_acc = pd.read_csv('accountancy.csv')


# In[46]:


df_acc.head()


# In[47]:


df_acc.groupby(['Executor', 'Type']).max()


# In[48]:


df.head()


# In[51]:


df.groupby(['attack_type', 'primary_attr']).id.nunique()


# In[52]:


df_algae = pd.read_csv('algae.csv')


# In[53]:


df_algae.head()


# In[54]:


mean_concentrations = df_algae.groupby('genus').aggregate({'alanin': 'mean', 'citrate': 'mean', 'glucose': 'mean', 'oleic_acid': 'mean'})


# In[55]:


mean_concentrations


# In[69]:


df_algae.query("genus == 'Fucus'")['alanin'].agg(['min', 'mean', 'max']).round(2)


# In[70]:


df_algae.groupby('genus').aggregate({'alanin': ['min', 'mean', 'max']}).loc[['Fucus']]


# In[71]:


df_algae.groupby('genus').agg(['min', 'mean', 'max']).loc['Fucus', 'alanin'].round(2)


# In[72]:


df_algae


# In[73]:


df_algae.groupby('group').aggregate({'glucose': lambda x: max(x) - min(x) })

