#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8


# In[24]:


data = pd.read_csv('bikes_rent.csv')


# In[25]:


data.head()


# In[26]:


data.shape


# In[27]:


data.info()


# In[28]:


data.describe()


# In[38]:


sns.violinplot(data['season'], data['cnt']);


# In[30]:


data['cnt'].hist(figsize=(10, 8));


# In[31]:


X_df, y_series = data.drop('cnt', axis=1), data['cnt']


# In[32]:


X_df.shape


# In[33]:


plt.figure(figsize=(12, 8))
for i, col in enumerate(X_df.columns):
    plt.subplot(4, 3, i + 1)
    plt.scatter(X_df[col], y_series)
    plt.title(col);


# In[43]:


sns.heatmap(data.corr(), annot=True, fmt='.1f');


# In[40]:


plt.scatter(data.mnth, data.cnt);


# In[41]:


plt.scatter(data.temp, data.cnt);


# In[74]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[65]:


linreg = LinearRegression()
lasso = Lasso(random_state=17)
ridge = Ridge(random_state=17)
lasso_cv = LassoCV(random_state=17, cv=5)
ridge_cv = RidgeCV()


# In[49]:


X, y = data.drop(['cnt'], axis=1).values, data['cnt'].values


# In[50]:


train_part_size = int(0.7 * X.shape[0])

X_train, X_valid =X[:train_part_size, :], X[train_part_size:, :]
y_train, y_valid =y[:train_part_size], y[train_part_size:]


# In[51]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


# In[52]:


linreg.fit(X_train_scaled, y_train)


# In[54]:


np.sqrt(mean_squared_error(y_valid, linreg.predict(X_valid_scaled)))


# In[57]:


pd.DataFrame(linreg.coef_, data.columns[: -1], columns=['coef']).sort_values(by='coef', ascending=False)


# In[79]:


def train_validate_report(model, X_train_scaled, y_train, X_valid_scaled, y_valid, feature_names, forest=False):
    '''
    For linear models and regression trees
    '''
    
    model.fit(X_train_scaled, y_train)
    
    
    print('MSE=%0.2f' % np.sqrt(mean_squared_error(y_valid, model.predict(X_valid_scaled))))
    print('Model coefficients:')
    coef = model.feature_importances_ if forest else model.coef_
    coef_name = 'Importance' if forest else 'Coef'
    print(pd.DataFrame(coef, feature_names, columns=[coef_name]).sort_values(by=coef_name, ascending=False))


# In[63]:


train_validate_report(lasso, X_train_scaled, y_train, 
                      X_valid_scaled, y_valid, 
                      feature_names=data.columns[: -1])


# In[66]:


train_validate_report(lasso_cv, X_train_scaled, y_train, 
                      X_valid_scaled, y_valid, 
                      feature_names=data.columns[: -1])


# In[67]:


train_validate_report(ridge, X_train_scaled, y_train, 
                      X_valid_scaled, y_valid, 
                      feature_names=data.columns[: -1])


# In[68]:


train_validate_report(ridge_cv, X_train_scaled, y_train, 
                      X_valid_scaled, y_valid, 
                      feature_names=data.columns[: -1])


# In[85]:


forest = RandomForestRegressor(random_state=17, n_estimators=500)


# In[86]:


train_validate_report(forest, X_train_scaled, y_train, 
                      X_valid_scaled, y_valid, 
                      feature_names=data.columns[: -1], forest=True)


# In[87]:


train_validate_report(forest, X_train, y_train, 
                      X_valid, y_valid, 
                      feature_names=data.columns[: -1], forest=True)

