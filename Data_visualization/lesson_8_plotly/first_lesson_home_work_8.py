#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pokemon = pd.read_csv("Pokemon.csv")
pokemon.head(3)


# In[2]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

iplot([go.Scatter(x=pokemon['Attack'], y=pokemon['Defense'], mode='markers')])

