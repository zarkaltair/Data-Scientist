#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# ## Загрузка и подготовка данных
# 
# Посмотрим на примере данных по оттоку клиентов из телеком-компании.

# In[4]:


df = pd.read_csv('telecom_churn.csv')


# In[5]:


df.head()


# **Штаты просто занумеруем, а признаки International plan (наличие международного роуминга), Voice mail plan (наличие голосовой почтыы) и целевой Churn сделаем бинарными.**

# In[6]:


state_enc = LabelEncoder()
df['State'] = state_enc.fit_transform(df['State'])
df['International plan'] = (df['International plan'] == 'Yes').astype('int')
df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype('int')
df['Churn'] = (df['Churn']).astype('int')


# **Разделим данные на обучающую и тестовую выборки в отношении 7:3. Создадим соотв. объекты DMAtrix.**

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'],
                                                    test_size=0.3, random_state=42)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


# **Зададим параметры Xgboost.**

# In[8]:


params = {
    'objective':'binary:logistic',
    'max_depth':3,
    'silent':1,
    'eta':0.5
}

num_rounds = 10


# **Будем отслеживать качество модели и на обучающей выборке, и на валидационной.**

# In[9]:


watchlist  = [(dtest,'test'), (dtrain,'train')]


# ## Использование встроенных метрик 
# В Xgboost реализованы большинство популярных метрик для классификации, регрессии и ранжирования:
# 
# - `rmse` - [root mean square error](https://www.wikiwand.com/en/Root-mean-square_deviation)
# - `mae` - [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error?oldformat=true)
# - `logloss` - [negative log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function?oldformat=true)
# - `error` (по умолчанию) - доля ошибок в бинарной классификации
# - `merror` - доля ошибок в классификации на несколько классов
# - `auc` - [area under curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic?oldformat=true)
# - `ndcg` - [normalized discounted cumulative gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain?oldformat=true)
# - `map` - [mean average precision](https://en.wikipedia.org/wiki/Information_retrieval?oldformat=true)

# In[10]:


xgb_model = xgb.train(params, dtrain, num_rounds, watchlist)


# **Чтоб отслеживать log_loss, просто добавим ее в словарь params.**

# In[11]:


params['eval_metric'] = 'logloss'
xgb_model = xgb.train(params, dtrain, num_rounds, watchlist)


# **Можно отслеживать сразу несколько метрик.**

# In[12]:


params['eval_metric'] = ['logloss', 'auc']
xgb_model = xgb.train(params, dtrain, num_rounds, watchlist)


# ## Создание собственной метрики качества
# 
# **Чтобы создать свою метрику качества, достаточно определить функцию, принимающую 2 аргумента: вектор предсказанных вероятностей и объект `DMatrix` с истинными метками.  
# В этом примере функция вернет просто число объектов, на которых классификатор ошибся, когла относил к классу 1 при превышении предсказанной вероятности класса 1 порога 0.5. 
# Далее передаем эту функцию в xgb.train (параметр feval), если метрика тем лучше, чем меньше, надо дополнительно указать `maximize=False`.**
# 

# In[13]:


# custom evaluation metric
def misclassified(pred_probs, dmatrix):
    labels = dmatrix.get_label() # obtain true labels
    preds = pred_probs > 0.5 # obtain predicted values
    return 'misclassified', np.sum(labels != preds)


# In[14]:


xgb_model = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False)


# **С помощью параметра evals_result можно сохранить значения метрик по итерациям.**

# In[15]:


evals_result = {}
xgb_model = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False, 
                      evals_result=evals_result)


# In[16]:


evals_result


# ## Ранняя остановка
# **Ранняя остановка используется для того, чтобы прекратить обучение модели, если ошибка за несколько итераций не уменьшилась.**

# In[17]:


params['eval_metric'] = 'error'
num_rounds = 1500

xgb_model = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)


# In[18]:


print("Booster best train score: {}".format(xgb_model.best_score))
print("Booster best iteration: {}".format(xgb_model.best_iteration))


# ## Кросс-валидация с Xgboost
# **Продемонстрируем функцию xgboost.cv.**

# In[19]:


num_rounds = 10
hist = xgb.cv(params, dtrain, num_rounds, nfold=10, metrics={'error'}, seed=42)
hist


# Замечания:
# 
# - по умолчанию на выходе DataFrame (можно поменять параметр `as_pandas`),
# - метрики передатся как параметр (можно и несколько),
# - можно использовать и свои метрики (параметры `feval` и `maximize`),
# - можно также использовать раннюю остановку ( `early_stopping_rounds`)
