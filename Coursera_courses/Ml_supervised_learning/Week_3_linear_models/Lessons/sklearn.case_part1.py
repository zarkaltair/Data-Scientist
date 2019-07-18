#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## Bike Sharing Demand
# Задача на kaggle: https://www.kaggle.com/c/bike-sharing-demand
# 
# По историческим данным о прокате велосипедов и погодным условиям необходимо оценить спрос на прокат велосипедов.
# 
# В исходной постановке задачи доступно 11 признаков: https://www.kaggle.com/c/prudential-life-insurance-assessment/data
# 
# В наборе признаков присутсвуют вещественные, категориальные, и бинарные данные. 
# 
# Для демонстрации используется обучающая выборка из исходных данных train.csv, файлы для работы прилагаются.

# ### Библиотеки

# In[19]:


from sklearn import linear_model, metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Загрузка данных

# In[6]:


raw_data = pd.read_csv('bike_sharing_demand.csv', header=0, sep=',')


# In[7]:


raw_data.head()


# ***datetime*** - hourly date + timestamp  
# 
# ***season*** -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# 
# ***holiday*** - whether the day is considered a holiday
# 
# ***workingday*** - whether the day is neither a weekend nor holiday
# 
# ***weather*** - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
#     
# ***temp*** - temperature in Celsius
# 
# ***atemp*** - "feels like" temperature in Celsius
# 
# ***humidity*** - relative humidity
# 
# ***windspeed*** - wind speed
# 
# ***casual*** - number of non-registered user rentals initiated
# 
# ***registered*** - number of registered user rentals initiated
# 
# ***count*** - number of total rentals

# In[8]:


raw_data.shape


# In[9]:


raw_data.isnull().values.any()


# ### Предобработка данных

# #### Типы признаков

# In[10]:


raw_data.info()


# In[11]:


raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)


# In[12]:


raw_data['month'] = raw_data.datetime.apply(lambda x : x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x : x.hour)


# In[13]:


raw_data.head()


# #### Обучение и отложенный тест

# In[14]:


train_data = raw_data.iloc[:-1000, :]
hold_out_test_data = raw_data.iloc[-1000:, :]


# In[15]:


raw_data.shape, train_data.shape, hold_out_test_data.shape


# In[16]:


print('train period from {} to {}'.format(train_data.datetime.min(), train_data.datetime.max()))
print('evaluation period from {} to {}'.format(hold_out_test_data.datetime.min(), hold_out_test_data.datetime.max()))


# #### Данные и целевая функция

# In[17]:


#обучение
train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count'], axis = 1)


# In[18]:


#тест
test_labels = hold_out_test_data['count'].values
test_data = hold_out_test_data.drop(['datetime', 'count'], axis = 1)


# #### Целевая функция на обучающей выборке и на отложенном тесте

# In[20]:


plt.figure(figsize=(16, 6))

plt.subplot(1,2,1)
plt.hist(train_labels)
plt.title('train data')

plt.subplot(1,2,2)
plt.hist(test_labels)
plt.title('test data');


# #### Числовые признаки

# In[21]:


numeric_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'month', 'hour']


# In[22]:


train_data = train_data[numeric_columns]
test_data = test_data[numeric_columns]


# In[23]:


train_data.head()


# In[24]:


test_data.head()


# ### Модель

# In[25]:


regressor = linear_model.SGDRegressor(random_state=0)


# In[26]:


regressor.fit(train_data, train_labels)
metrics.mean_absolute_error(test_labels, regressor.predict(test_data))


# In[27]:


test_labels[:10]


# In[28]:


regressor.predict(test_data)[:10]


# In[29]:


regressor.coef_


# ### Scaling

# In[30]:


from sklearn.preprocessing import StandardScaler


# In[31]:


#создаем стандартный scaler
scaler = StandardScaler()
scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[32]:


regressor.fit(scaled_train_data, train_labels)
metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data))


# In[34]:


test_labels[:10]


# In[35]:


regressor.predict(scaled_test_data)[:10]


# ### Подозрительно хорошо?

# In[36]:


regressor.coef_


# In[37]:


map(lambda x : round(x, 2), regressor.coef_)


# In[38]:


train_data.head()


# In[39]:


train_labels[:10]


# In[40]:


np.all(train_data.registered + train_data.casual == train_labels)


# In[42]:


train_data.drop(['casual', 'registered'], axis=1, inplace=True)
test_data.drop(['casual', 'registered'], axis=1, inplace=True)


# In[43]:


scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[44]:


regressor.fit(scaled_train_data, train_labels)
metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data))


# In[46]:


map(lambda x : round(x, 2), regressor.coef_)


# ### Pipeline

# In[47]:


from sklearn.pipeline import Pipeline


# In[48]:


#создаем pipeline из двух шагов: scaling и классификация
pipeline = Pipeline(steps = [('scaling', scaler), ('regression', regressor)])


# In[49]:


pipeline.fit(train_data, train_labels)
metrics.mean_absolute_error(test_labels, pipeline.predict(test_data))


# ### Подбор параметров

# In[50]:


pipeline.get_params().keys()


# In[60]:


parameters_grid = {
    'regression__loss' : ['huber', 'epsilon_insensitive', 'squared_loss', ],
    'regression__max_iter' : [3, 5, 10, 50, 100], 
    'regression__penalty' : ['l1', 'l2', 'none'],
    'regression__alpha' : [0.0001, 0.01],
    'scaling__with_mean' : [0., 0.5],
}


# In[63]:


grid_cv = GridSearchCV(pipeline, parameters_grid, scoring='neg_mean_squared_error', cv=4)


# In[64]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(train_data, train_labels)')


# In[65]:


print(grid_cv.best_score_)
print(grid_cv.best_params_)


# ### Оценка по отложенному тесту

# In[66]:


metrics.mean_absolute_error(test_labels, grid_cv.best_estimator_.predict(test_data))


# In[67]:


np.mean(test_labels)


# In[68]:


test_predictions = grid_cv.best_estimator_.predict(test_data)


# In[69]:


test_labels[:10]


# In[70]:


test_predictions[:10]


# In[71]:


plt.figure(figsize=(16, 6))

plt.subplot(1,2,1)
plt.grid(True)
plt.scatter(train_labels, pipeline.predict(train_data), alpha=0.5, color = 'red')
plt.scatter(test_labels, pipeline.predict(test_data), alpha=0.5, color = 'blue')
plt.title('no parameters setting')
plt.xlim(-100,1100)
plt.ylim(-100,1100)

plt.subplot(1,2,2)
plt.grid(True)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
plt.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
plt.title('grid search')
plt.xlim(-100,1100)
plt.ylim(-100,1100);

