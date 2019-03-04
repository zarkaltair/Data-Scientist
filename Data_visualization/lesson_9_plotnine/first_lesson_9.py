#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
reviews.head(3)


# In[4]:


from plotnine import *
top_wines = reviews[reviews['variety'].isin(reviews['variety'].value_counts().head(5).index)]


# In[5]:


df = top_wines.head(1000).dropna()

(ggplot(df)
 + aes('points', 'price')
 + geom_point())


# In[6]:


df = top_wines.head(1000).dropna()

(
    ggplot(df)
        + aes('points', 'price')
        + geom_point()
        + stat_smooth()
)


# In[7]:


df = top_wines.head(1000).dropna()

(
    ggplot(df)
        + geom_point()
        + aes(color='points')
        + aes('points', 'price')
        + stat_smooth()
)


# In[8]:


df = top_wines.head(1000).dropna()

(ggplot(df)
     + aes('points', 'price')
     + aes(color='points')
     + geom_point()
     + stat_smooth()
     + facet_wrap('~variety')
)


# In[9]:


(ggplot(df)
 + geom_point(aes('points', 'price'))
)


# In[10]:


(ggplot(df, aes('points', 'price'))
 + geom_point()
)


# In[11]:


(ggplot(top_wines)
     + aes('points')
     + geom_bar()
)


# In[12]:


(ggplot(top_wines)
     + aes('points', 'variety')
     + geom_bin2d(bins=20)
)


# In[13]:


(ggplot(top_wines)
         + aes('points', 'variety')
         + geom_bin2d(bins=20)
         + coord_fixed(ratio=1)
         + ggtitle("Top Five Most Common Wine Variety Points Awarded")
)

