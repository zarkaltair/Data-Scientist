#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set(rc={'figure.figsize': (12, 6)})


# In[4]:


event_data = pd.read_csv('event_data_train.csv')


# In[5]:


event_data.head(10)


# In[6]:


event_data.action.unique()


# In[7]:


event_data['date'] = pd.to_datetime(event_data.timestamp, unit='s')


# In[8]:


event_data.head()


# In[9]:


event_data.dtypes


# In[10]:


event_data.date.min()


# In[11]:


event_data.date.max()


# In[12]:


event_data['day'] = event_data.date.dt.date


# In[13]:


event_data.head()


# In[14]:


event_data.groupby('day').user_id.nunique().plot();


# In[15]:


event_data[event_data.action == 'passed']     .groupby('user_id', as_index=False)     .agg({'step_id': 'count'})     .rename(columns={'step_id': 'passed_steps'}).passed_steps.hist();


# In[66]:


users_events_data = event_data.pivot_table(index='user_id', 
                       columns='action', 
                       values='step_id', 
                       aggfunc='count', 
                       fill_value=0).reset_index()


# In[67]:


users_events_data.head()


# In[17]:


submissions_data = pd.read_csv('submissions_data_train.csv')


# In[18]:


submissions_data.head()


# In[19]:


submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
submissions_data['day'] = submissions_data.date.dt.date


# In[65]:


users_scores = submissions_data.pivot_table(index='user_id', 
                       columns='submission_status', 
                       values='step_id', 
                       aggfunc='count', 
                       fill_value=0).reset_index()


# In[22]:


gap_data = event_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day'])     .groupby('user_id')['timestamp'].apply(list)     .apply(np.diff).values


# In[23]:


gap_data = pd.Series(np.concatenate(gap_data, axis=0))


# In[24]:


gap_data = gap_data / (24 * 60 * 60)


# In[25]:


gap_data[gap_data < 200].hist();


# In[26]:


gap_data.quantile(0.95)


# In[27]:


submissions_data.groupby('user_id', as_index=False).aggregate({'date': 'count'}).max()


# In[28]:


submissions_data.head()


# In[29]:


submissions_data_corr = submissions_data[submissions_data.submission_status == 'correct']


# In[30]:


ss = submissions_data_corr.groupby('user_id', as_index=False).aggregate({'submission_status': 'count'})


# In[31]:


event_data.tail()


# In[43]:


users_data = event_data.groupby('user_id', as_index=False)     .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})


# In[49]:


now = 1526772811
drop_out_threshold = 2592000


# In[52]:


users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold


# In[53]:


users_data.head()


# In[54]:


users_scores.head()


# In[59]:


users_data = users_data.merge(users_scores, on='user_id', how='outer')


# In[61]:


users_data = users_data.fillna(0)


# In[68]:


users_data = users_data.merge(users_events_data, on='user_id', how='outer')


# In[75]:


users_days = event_data.groupby('user_id').day.nunique().to_frame().reset_index().head()


# In[78]:


users_data = users_data.merge(users_days, on='user_id', how='outer')


# In[79]:


users_data.head()


# In[80]:


users_data.user_id.nunique()


# In[81]:


event_data.user_id.nunique()


# In[82]:


users_data['passed_course'] = users_data.passed > 170


# In[83]:


users_data.head()


# In[84]:


users_data.groupby('passed_course').count()


# In[85]:


100 * 1425 / 17809

