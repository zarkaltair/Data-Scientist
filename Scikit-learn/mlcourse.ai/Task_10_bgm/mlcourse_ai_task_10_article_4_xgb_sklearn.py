#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# ## Загрузка и подготовка данных
# 
# Посмотрим на примере данных по оттоку клиентов из телеком-компании.

# In[2]:


df = pd.read_csv('telecom_churn.csv')


# In[3]:


df.head()


# **Штаты просто занумеруем, а признаки International plan (наличие международного роуминга), Voice mail plan (наличие голосовой почтыы) и целевой Churn сделаем бинарными.**

# In[4]:


state_enc = LabelEncoder()
df['State'] = state_enc.fit_transform(df['State'])
df['International plan'] = (df['International plan'] == 'Yes').astype('int')
df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype('int')
df['Churn'] = (df['Churn']).astype('int')


# **Разделим данные на обучающую и тестовую выборки в отношении 7:3.**

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'],
                                                    test_size=0.3, stratify=df['Churn'], random_state=17)


# ### Инициализация параметров
# 
# - бинарная классификация (`'objective':'binary:logistic'`)
# - ограничим глубину деревьев (`'max_depth':3`)
# - не хотим лишнего вывода (`'silent':1`)
# - проведем 10 итераций бустинга
# - шаг градиентного спуска довольно большой (`'eta':1`) - алгоритм будет обучаться быстро и "агрессивно" (лучше результаты будут, если уменьшить eta и увеличить число итераций)
# 

# In[10]:


params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 1.0,
    'silent': 1,
    'n_estimators': 50
}


# ### Обучение классификатора
# Тут мы просто передаем слоавть параметров, данные и число итераций.

# In[11]:


xgb_model = XGBClassifier(**params).fit(X_train, y_train)


# ### Прогнозы для тестовой выборки

# In[12]:


preds_prob = xgb_model.predict(X_test)


# **Посчитаем долю правильных ответов алгоритма на тестовой выборке.**

# In[13]:


predicted_labels = preds_prob > 0.5
print("Accuracy and F1 on the test set are: {} and {}".format(
    round(accuracy_score(y_test, predicted_labels), 3),
    round(f1_score(y_test, predicted_labels), 3)))

