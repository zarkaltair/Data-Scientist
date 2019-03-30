#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb


# **Загрузим данные и осуществим минимальную предобработку.**

# In[2]:


df = pd.read_csv('telecom_churn.csv')


# In[3]:


df.head()


# **Штаты просто занумеруем (хотя можно и лучше поступить), а признаки International plan (наличие международного роуминга), Voice mail plan (наличие голосовой почтыы) и целевой Churn сделаем бинарными.**

# In[4]:


state_enc = LabelEncoder()
df['State'] = state_enc.fit_transform(df['State'])
df['International plan'] = (df['International plan'] == 'Yes').astype('int')
df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype('int')
df['Churn'] = (df['Churn']).astype('int')


# **Разделим данные на обучающую и тестовую выборки в отношении 7:3. Инициализируем соотв. объекты DMatrix dtrain и dtest.**

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'],
                                                    test_size=0.3, stratify=df['Churn'], random_state=17)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


# **Посмотрим на статистику полученных объектов:**

# In[6]:


print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))


# In[7]:


print("Train mean target: ")
print(np.mean(dtrain.get_label()))

print("\nTest mean target: ")
print(np.mean(dtest.get_label()))


# ### Инициализация параметров
# 
# - бинарная классификация (`'objective':'binary:logistic'`)
# - ограничим глубину деревьев (`'max_depth':3`)
# - не хотим лишнего вывода (`'silent':1`)
# - проведем 50 итераций бустинга
# - шаг градиентного спуска довольно большой (`'eta':1`) - алгоритм будет обучаться быстро и "агрессивно" (лучше результаты будут, если уменьшить eta и увеличить число итераций)
# 

# In[8]:


params = {
    'objective':'binary:logistic',
    'max_depth': 3,
    'silent': 1,
    'eta': 1
}

num_rounds = 50


# ### Обучение классификатора
# Тут мы просто передаем слоавть параметров, данные и число итераций.

# In[9]:


xgb_model = xgb.train(params, dtrain, num_rounds)


# **С помощью `watchlist` отслеживать качество алгоритма на тестовой выборке для каждой итерации.**

# In[10]:


watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only
xgb_model = xgb.train(params, dtrain, num_rounds, watchlist)


# ### Прогнозы для тестовой выборки

# In[11]:


preds_prob = xgb_model.predict(dtest)


# **Посчитаем долю правильных ответов алгоритма на тестовой выборке.**

# In[12]:


predicted_labels = preds_prob > 0.5
print("Accuracy and F1 on the test set are: {} and {}".format(
    round(accuracy_score(y_test, predicted_labels), 3),
    round(f1_score(y_test, predicted_labels), 3)))

