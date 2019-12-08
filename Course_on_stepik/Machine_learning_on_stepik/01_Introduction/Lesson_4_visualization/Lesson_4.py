#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


students_performance = pd.read_csv('StudentsPerformance.csv')


# In[3]:


students_performance.head(3)


# In[4]:


students_performance['math score'].hist();


# In[5]:


students_performance.plot.scatter(x='math score', y='reading score');


# In[11]:


ax = sns.lmplot(x='math score', y='reading score', hue='gender', data=students_performance, fit_reg=False)
ax.set_xlabels('Math score')
ax.set_ylabels('Reading score');


# In[12]:


df = pd.read_csv('income.csv')


# In[13]:


df.head(3)


# In[15]:


df.income.plot();


# In[17]:


df.plot(kind='line');


# In[19]:


plt.plot(df.index, df.income);


# In[21]:


df.plot();


# In[22]:


sns.lineplot(data=df);


# In[24]:


df['income'].plot();


# In[25]:


sns.lineplot(x=df.index, y=df.income);


# In[29]:


dff = pd.read_csv('dataset_209770_6.txt', sep=' ')


# In[31]:


dff.head(3)


# In[32]:


dff.plot.scatter(x='x', y='y');


# In[39]:


df_genome = pd.read_csv('genome_matrix.csv', index_col=0)


# In[40]:


df_genome.head(3)


# In[43]:


g = sns.heatmap(data=df_genome, cmap='viridis')
g.xaxis.set_ticks_position('top')
g.xaxis.set_tick_params(rotation=90)


# In[46]:


df_d2 = pd.read_csv('dota_hero_stats.csv', index_col=0)


# In[47]:


df_d2.head()


# In[70]:


df_d2.roles.str.split(',').apply(len).mode()


# In[72]:


sns.distplot([x.count(',') + 1 for x in df_d2.roles], bins=15);


# In[2]:


df_iris = pd.read_csv('iris.csv', index_col=0)
df_iris.head()


# In[11]:


sns.distplot(df_iris["sepal length"] , color="skyblue", label="Sepal Length")
sns.distplot(df_iris["sepal width"] , color="red", label="Sepal Width")
sns.distplot(df_iris["petal length"] , color="green", label="Petal Length")
sns.distplot(df_iris["petal width"] , color="blue", label="Petal Width");


# In[7]:


f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot(df_iris["sepal length"] , color="skyblue", ax=axes[0, 0])
sns.distplot(df_iris["sepal width"] , color="olive", ax=axes[0, 1])
sns.distplot(df_iris["petal length"] , color="gold", ax=axes[1, 0])
sns.distplot(df_iris["petal width"] , color="teal", ax=axes[1, 1]);


# In[13]:


sns.violinplot(x=df_iris['petal length']);


# In[15]:


sns.pairplot(df_iris, hue="species");

