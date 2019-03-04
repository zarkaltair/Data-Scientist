#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from plotnine import *
pokemon = pd.read_csv("Pokemon.csv", index_col=0)                        .rename(columns=lambda x: x.replace(" ", "_"))
pokemon.head(3)


# In[5]:


(ggplot(pokemon)
 + aes('Attack', 'Defense')
 + geom_point())


# In[6]:


(
    ggplot(pokemon, aes('Attack', 'Defense', color='Legendary'))
        + geom_point()
        + ggtitle("Pokemon Attack and Defense by Legendary Status")
)


# In[7]:


(
    ggplot(pokemon, aes('Attack'))
        + geom_histogram(bins=20)
) + facet_wrap('~Generation')

