#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
#графики в svg выглядят более четкими
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# In[2]:


df = pd.read_csv('telecom_churn.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df['Churn'].value_counts()


# In[7]:


df['Churn'].value_counts().plot(kind='bar', label='Churn')
plt.legend()
plt.title('Распределение оттока колиентов')


# In[8]:


corr_matrix = df.drop(['State', 'International plan', 'Voice mail plan', 'Area code'], axis=1).corr()


# In[11]:


sns.heatmap(corr_matrix);


# In[16]:


features = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan','Area code',
                                       'Total day charge', 'Total eve charge', 'Total night charge',
                                       'Total intl charge', 'Churn']))
df[features].hist(figsize=(12,10));


# In[ ]:


sns.pairplot(df[features + ['Churn']], hue='Churn');


# In[17]:


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 10))

for idx, feat in enumerate(features):
    sns.boxplot(x='Churn', y=feat, data=df, ax=axes[idx / 4, idx % 4])
    axes[idx / 4, idx % 4].legend()
    axes[idx / 4, idx % 4].set_xlabel('Churn')
    axes[idx / 4, idx % 4].set_ylabel(feat);


# In[19]:


_, axes = plt.subplots(1, 2, sharey=True, figsize=(12,6))

sns.boxplot(x='Churn', y='Total day minutes', data=df, ax=axes[0]);
sns.violinplot(x='Churn', y='Total day minutes', data=df, ax=axes[1]);


# In[20]:


sns.countplot(x='Customer service calls', hue='Churn', data=df);


# In[24]:


_, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0]);
sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1]);


# In[27]:


df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T


# In[28]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# In[29]:


X = df.drop(['Churn', 'State'], axis=1)
X['International plan'] = pd.factorize(X['International plan'])[0]
X['Voice mail plan'] = pd.factorize(X['Voice mail plan'])[0]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[30]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(random_state=17)\ntsne_representation = tsne.fit_transform(X_scaled)')


# In[31]:


plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1]);


# In[32]:


plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=df['Churn'].map({0: 'blue', 1: 'orange'}));


# In[36]:


_, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

axes[0].scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=df['International plan'].map({'Yes': 'blue', 'No': 'orange'}));
axes[1].scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=df['Voice mail plan'].map({'Yes': 'blue', 'No': 'orange'}));
axes[0].set_title('International plan');
axes[1].set_title('Voice mail plan');


# ## home work

# In[37]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# In[38]:


df = pd.read_csv('howpop_train.csv')
df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




