#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


stock = pd.read_csv('amzn_stock.csv', index_col='Date', parse_dates=True)
stock.head()


# In[5]:


stock.index


# In[18]:


stock['2010-02': '2010-03'].head()


# In[16]:


stock.resample('1w').mean().head()


# In[17]:


stock.rolling(3, min_periods=1).mean().head()


# In[15]:


stock.expanding().mean().head()


# In[14]:


stock.ewm(alpha=0.7).mean().head()


# In[20]:


stock['Open'].plot();


# In[21]:


ns = stock['Open'].rolling(10, min_periods=1).mean()


# In[23]:


stock['Open'].plot()
ns.plot();


# In[24]:


stock.index.weekday


# In[25]:


stock.index.weekday_name


# In[26]:


stock.index.weekday_name.value_counts()


# In[27]:


stock.index.dayofyear


# In[29]:


stock.index.weekofyear


# In[30]:


stock.index.year


# In[32]:


stock.index.month_name()


# In[33]:


np.mean(stock.index.dayofyear)

