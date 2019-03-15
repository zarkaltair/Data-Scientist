#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np


# In[34]:


df = pd.read_csv('telecom_churn.csv')
df.head(3)


# In[4]:


df.shape


# In[5]:


df.columns


# In[7]:


df.info()


# In[8]:


df['Churn'] = df['Churn'].astype('int64')


# In[9]:


df.describe()


# In[10]:


df.describe(include=['object', 'bool'])


# In[12]:


df['Churn'].value_counts()


# In[14]:


df['Area code'].value_counts()


# In[13]:


df['Area code'].value_counts(normalize=True)


# In[17]:


df.sort_values(by='Total day charge', ascending=False).head()


# In[18]:


df.sort_values(by=['Churn', 'Total day charge'], ascending=[True, False]).head()


# In[19]:


df['Churn'].mean()


# In[20]:


df[df['Churn'] == 1].mean()


# In[21]:


df[df['Churn'] == 1]['Total day minutes'].mean()


# In[24]:


df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max()


# In[25]:


df.loc[0: 5, 'State': 'Area code']


# In[26]:


df.iloc[0: 5, 0: 3]


# In[27]:


df[-1:]


# In[30]:


df.apply(np.max)


# In[35]:


d = {'No': False, 'Yes': True}
df['International plan'] = df['International plan'].map(d)
df.head()


# In[36]:


df.replace({'Voice mail plan': d})
df.head()


# In[39]:


columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']
df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])


# In[41]:


df.groupby(['Churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max])


# In[42]:


pd.crosstab(df['Churn'], df['International plan'])


# In[44]:


pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)


# In[46]:


df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'], 
               ['Area code'], aggfunc='mean').head(10)


# In[48]:


total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns), column='Total calls', value=total_calls)
df.head()


# In[50]:


df['Total charge'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl charge']
df.head()


# In[69]:


# df = df.drop(['Total charge'], axis=1, inplace=True)
# df.drop([1, 2]).head()


# ## home work

# In[74]:


import pandas as pd


# In[87]:


data = pd.read_csv('adult.data.csv')
data.head(3)


# In[88]:


data.shape


# In[79]:


data['sex'].value_counts()


# In[82]:


data[data['sex'] == 'Female']['age'].mean()


# In[90]:


data['native-country'].value_counts(normalize=True)


# In[92]:


data[data['salary'] == '<=50K']['age'].mean()


# In[95]:


data[data['salary'] != '<=50K']['age'].mean()


# In[96]:


data[data['salary'] == '<=50K']['age'].std()


# In[97]:


data[data['salary'] != '<=50K']['age'].std()


# In[101]:


data[data['salary'] != '<=50K']['education'].value_counts()


# In[107]:


data.groupby(['race'])['age'].describe(percentiles=[])


# In[106]:


data[data['race'] == 'Amer-Indian-Eskimo']['age'].max()


# In[112]:


data[data['salary'] == '<=50K']['marital-status'].value_counts(normalize=True)


# In[113]:


data[data['salary'] != '<=50K']['marital-status'].value_counts(normalize=True)


# In[114]:


data['hours-per-week'].max()


# In[116]:


data[data['hours-per-week'] == 99]['age'].count()


# In[121]:


pd.crosstab(data['hours-per-week'], data['salary'])
25 / (60 + 25)


# In[131]:


data.pivot_table(['hours-per-week'], 
               ['salary'], aggfunc='mean')


# In[136]:


data[(data['native-country'] == 'Japan') & (data['salary'] == '<=50K')]['hours-per-week'].mean()


# In[137]:


data[(data['native-country'] == 'Japan') & (data['salary'] != '<=50K')]['hours-per-week'].mean()

