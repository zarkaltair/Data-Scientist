#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
reviews.head()


# In[3]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# In[4]:


iplot([go.Histogram2dContour(x=reviews.head(500)['points'], 
                             y=reviews.head(500)['price'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# In[5]:


df = reviews.assign(n=0).groupby(['points', 'price'])['n'].count().reset_index()
df = df[df["price"] < 100]
v = df.pivot(index='price', columns='points', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=v)])


# In[6]:


df = reviews['country'].replace("US", "United States").value_counts()

iplot([go.Choropleth(
    locationmode='country names',
    locations=df.index.values,
    text=df.index,
    z=df.values
)])

