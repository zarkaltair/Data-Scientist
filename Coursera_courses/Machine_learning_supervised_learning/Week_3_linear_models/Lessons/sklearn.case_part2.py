#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## Bike Sharing Demand
# Задача на kaggle: https://www.kaggle.com/c/bike-sharing-demand
# 
# По историческим данным о прокате велосипедов и погодных условиях необходимо спрогнозировтаь спрос на прокат велосипедов.
# 
# В исходной постановке задачи доступно 11 признаков: https://www.kaggle.com/c/prudential-life-insurance-assessment/data
# 
# В наборе признаков присутсвуют вещественные, категориальные, и бинарные данные. 
# 
# Для демонстрации используется обучающая выборка из исходных данных train.csv, файлы для работы прилагаются.

# ### Библиотеки

# In[3]:


from sklearn import linear_model, metrics, pipeline, preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Загрузка данных

# In[5]:


raw_data = pd.read_csv('bike_sharing_demand.csv', header=0, sep=',')


# In[6]:


raw_data.head()


# ### Предобработка данных

# #### Обучение и отложенный тест

# In[7]:


raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)


# In[8]:


raw_data['month'] = raw_data.datetime.apply(lambda x : x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x : x.hour)


# In[9]:


train_data = raw_data.iloc[:-1000, :]
hold_out_test_data = raw_data.iloc[-1000:, :]


# In[10]:


raw_data.shape, train_data.shape, hold_out_test_data.shape


# In[11]:


#обучение
train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)


# In[12]:


#тест
test_labels = hold_out_test_data['count'].values
test_data = hold_out_test_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)


# In[13]:


binary_data_columns = ['holiday', 'workingday']
binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)


# In[14]:


print(binary_data_columns)
print(binary_data_indices)


# In[15]:


categorical_data_columns = ['season', 'weather', 'month'] 
categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)


# In[16]:


print(categorical_data_columns)
print(categorical_data_indices)


# In[17]:


numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)


# In[18]:


print(numeric_data_columns)
print(numeric_data_indices)


# ### Pipeline

# In[20]:


from sklearn.linear_model import SGDRegressor


# In[21]:


regressor = SGDRegressor(random_state=0, max_iter=3, loss='squared_loss', penalty='l2')


# In[22]:


estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', regressor)
    ]
)


# In[23]:


estimator.fit(train_data, train_labels)


# In[24]:


metrics.mean_absolute_error(test_labels, estimator.predict(test_data))


# ### Подбор параметров

# In[25]:


estimator.get_params().keys()


# In[26]:


parameters_grid = {
    'model_fitting__alpha' : [0.0001, 0.001, 0,1],
    'model_fitting__eta0' : [0.001, 0.05],
}


# In[27]:


grid_cv = GridSearchCV(estimator, parameters_grid, scoring = 'neg_mean_absolute_error', cv=4)


# In[28]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(train_data, train_labels)')


# In[29]:


print(grid_cv.best_score_)
print(grid_cv.best_params_)


# ### Оценка по отложенному тесту

# In[30]:


test_predictions = grid_cv.best_estimator_.predict(test_data)


# In[31]:


metrics.mean_absolute_error(test_labels, test_predictions)


# In[32]:


test_labels[:20]


# In[33]:


test_predictions[:20]


# In[34]:


plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
plt.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue');


# ### Другая модель

# In[35]:


from sklearn.ensemble import RandomForestRegressor


# In[37]:


regressor = RandomForestRegressor(random_state=0, max_depth=20, n_estimators=50)


# In[38]:


estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0, with_std = 1))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', regressor)
    ]
)


# In[39]:


estimator.fit(train_data, train_labels)


# In[40]:


metrics.mean_absolute_error(test_labels, estimator.predict(test_data))


# In[41]:


test_labels[:10]


# In[42]:


estimator.predict(test_data)[:10]


# In[43]:


plt.figure(figsize=(16, 6))

plt.subplot(1,2,1)
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
plt.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
plt.title('linear model')

plt.subplot(1,2,2)
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.scatter(train_labels, estimator.predict(train_data), alpha=0.5, color = 'red')
plt.scatter(test_labels, estimator.predict(test_data), alpha=0.5, color = 'blue')
plt.title('random forest model');

