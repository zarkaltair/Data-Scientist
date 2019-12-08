#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = {'type': ['A', 'A', 'B', 'B'], 'value': [10, 14, 12, 23]}
my_data = pd.DataFrame(data=data)
my_data


# In[2]:


my_stat = pd.read_csv('my_stat.csv')
my_stat.head()


# In[4]:


subset_1 = my_stat.iloc[0: 10, [0, 2]]


# In[5]:


subset_1


# In[6]:


bad_df = my_stat.index.isin([0, 4])
subset_df = my_stat[~bad_df]
subset_2 = subset_df.iloc[:, [1, 3]]


# In[8]:


subset_2.head()


# In[12]:


subset_1 = my_stat.loc[(my_stat['V1'] > 0) & (my_stat['V3'] == 'A')]


# In[26]:


subset_1.drop('V2', axis=1, inplace=True)
subset_1.drop('V4', axis=1, inplace=True)


# In[27]:


subset_1.shape


# In[40]:


subset_2 = my_stat.loc[(my_stat['V2'] != 10) | (my_stat['V4'] >= 1)]


# In[41]:


subset_2.drop('V1', axis=1, inplace=True)
subset_2.drop('V3', axis=1, inplace=True)


# In[42]:


subset_2.head()


# In[43]:


subset_2.shape


# In[44]:


my_stat['V5'] = my_stat.V1 + my_stat.V4


# In[45]:


my_stat.head()


# In[46]:


my_stat['V6'] = np.log(my_stat.V2)
my_stat.head()


# In[4]:


my_stat = my_stat.rename(index=str, columns={'V1': 'session_value', 'V2': 'group', 'V3': 'time', 'V4': 'n_users'})
my_stat.head()


# In[51]:


my_stat['session_value'] = my_stat['session_value'].fillna(0)


# In[ ]:


my_stat['n_users'] = my_stat['n_users'].median()


# In[67]:


med = my_stat.n_users.loc[my_stat['n_users'] > 0].median()


# In[88]:


indexes = my_stat.loc[my_stat['n_users'] <= 0].index


# In[92]:


for i in my_stat.n_users.loc[indexes]:
    my_stat.n_users.loc[i] = med


# In[101]:


my_stat.session_value.value_counts()


# In[112]:


df_na = pd.read_csv('my_stat_na.csv')
df_na.head()


# In[115]:


med = df_na.n_users.loc[df_na['n_users'] >= 0].median()
med


# In[110]:


df_na = df_na.fillna(0)


# In[111]:


df_na.count()


# In[116]:


indexes = df_na.loc[df_na['n_users'] < 0].index


# In[124]:


for i in indexes:
    df_na.n_users.loc[i] = med


# In[128]:


df_na['n_users'].head()


# In[12]:


mean_session_value_data = my_stat.groupby('group', as_index=False).agg({'session_value': 'mean'})


# In[13]:


mean_session_value_data.rename(index=str, columns={'session_value': 'mean_session_value'})

