#!/usr/bin/env python
# coding: utf-8

# # <center>Домашнее задание № 10 (демо)
# ## <center> Прогнозирование задержек вылетов
# 
# Ваша задача – побить единственный бенчмарк в [соревновании](https://www.kaggle.com/c/flight-delays-2017) на Kaggle Inclass. Подробных инструкций не будет, будет только тезисно описано, как получен этот бенчмарк. Конечно, с помощью Xgboost. Надеюсь, на данном этапе курса вам достаточно бросить полтора взгляда на данные, чтоб понять, что это тот тип задачи, в которой затащит Xgboost. Но проверьте еще Catboost.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


train = pd.read_csv('flight_delays_train.csv')
test = pd.read_csv('flight_delays_test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# Итак, надо по времени вылета самолета, коду авиакомпании-перевозчика, месту вылета и прилета и расстоянию между аэропортами вылета и прилета предсказать задержку вылета более 15 минут. В качестве простейшего бенчмарка возьмем логистическую регрессию и два признака, которые проще всего взять: `DepTime` и `Distance`. У такой модели результат – 0.68202 на LB. 

# In[5]:


X_train, y_train = train[['Distance', 'DepTime']].values, train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_test = test[['Distance', 'DepTime']].values

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, 
                                                                y_train, 
                                                                test_size=0.3, 
                                                                random_state=17)


# In[6]:


logit = LogisticRegression(random_state=17, solver='lbfgs')

logit.fit(X_train_part, y_train_part)
logit_valid_pred = logit.predict_proba(X_valid)[:, 1]

roc_auc_score(y_valid, logit_valid_pred)


# In[9]:


logit.fit(X_train, y_train)
logit_test_pred = logit.predict_proba(X_test)[:, 1]

pd.Series(logit_test_pred, 
          name='dep_delayed_15min').to_csv('submit_logit_2feat.csv', 
                                           index_label='id', 
                                           header=True)


# Как был получен бенчмарк в соревновании:
# - Признаки `Distance` и  `DepTime` брались без изменений
# - Создан признак "маршрут" из исходных `Origin` и `Dest`
# - К признакам `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier` и "маршрут" применено OHE-преобразование (`LabelBinarizer`)
# - Выделена отложенная выборка
# - Обучалась логистическая регрессия и градиентный бустинг (xgboost), гиперпараметры бустинга настраивались на кросс-валидации, сначала те, что отвечают за сложность модели, затем число деревьев фиксировалось равным 500 и настраивался шаг градиентного спуска
# - С помощью `cross_val_predict` делались прогнозы обеих моделей на кросс-валидации (именно предсказанные вероятности), настраивалась линейная смесь ответов логистической регрессии и градиентного бустинга вида $w_1 * p_{logit} + (1 - w_1) * p_{xgb}$, где $p_{logit}$ – предсказанные логистической регрессией вероятности класса 1, $p_{xgb}$ – аналогично. Вес $w_1$ подбирался вручную. 
# - В качестве ответа для тестовой выборки бралась аналогичная комбинация ответов двух моделей, но уже обученных на всей обучающей выборке.
# 
# Описанный план ни к чему не обязывает – это просто то, как решение получил автор задания. Возможно, мы не захотите следовать намеченному плану, а добавите, скажем, пару хороших признаков и обучите лес из тысячи деревьев.
# 
# Удачи!

# In[15]:


from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


# In[16]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# In[17]:


ctb = CatBoostClassifier(random_seed=17)


# In[21]:


train_df, y = train.drop('dep_delayed_15min', axis=1), train['dep_delayed_15min'].map({'Y': 1, 'N': 0})
train_df_part, valid_df, y_train_part, y_valid = train_test_split(train_df, y, 
                                                                  test_size=.3, stratify=y, 
                                                                  random_state=17)


# In[22]:


cat_features_idx = np.where(train_df_part.dtypes == 'object')[0].tolist()


# In[23]:


get_ipython().run_cell_magic('time', '', 'cv_scores = []\nfor train_idx, test_idx in skf.split(train_df_part, y_train_part):\n    cv_train_df, cv_valid_df = train_df_part.iloc[train_idx, :], train_df_part.iloc[test_idx, :]\n    y_cv_train, y_cv_valid = y_train_part.iloc[train_idx], y_train_part.iloc[test_idx]\n    \n    ctb.fit(cv_train_df, y_cv_train,\n        cat_features=cat_features_idx);\n    \n    cv_scores.append(roc_auc_score(y_cv_valid, ctb.predict_proba(cv_valid_df)[:, 1]))')


# In[24]:


np.mean(cv_scores)


# In[25]:


get_ipython().run_cell_magic('time', '', 'ctb.fit(train_df_part, y_train_part,\n        cat_features=cat_features_idx);')


# In[26]:


roc_auc_score(y_valid, ctb.predict_proba(valid_df)[:, 1])


# In[27]:


ctb.fit(train_df, y, cat_features=cat_features_idx)


# In[31]:


test['route'] = test.apply(lambda x: x['Origin'] + '-' + x['Dest'], 1)


# In[33]:


ctb_test_pred = ctb.predict_proba(test)[:, 1]


# In[34]:


pd.Series(ctb_test_pred, name='dep_delayed_15min').to_csv('submit_ctb.csv', 
                                                          index_label='id', 
                                                          header=True)


# In[ ]:





# In[ ]:





# In[7]:


train.info()


# In[8]:


train['route'] = train.apply(lambda x: x['Origin'] + '-' + x['Dest'], 1)


# In[4]:


dummies = ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'route']
train_dummies = pd.get_dummies(train[dummies])
train_dummies.head()


# In[ ]:





# In[5]:


train.head()


# In[6]:


X = train.drop(['dep_delayed_15min'], axis=1)


# In[7]:


y = train['dep_delayed_15min'].map({'Y': 1, 'N': 0})


# In[8]:


num_var = ['DepTime', 'Distance']
X = pd.concat([train[num_var], train_dummies], axis=1)
X.head()


# In[9]:


split = int(X.shape[0] * 0.7)


# In[10]:


X_train = X.iloc[:split, :]
X_valid = X.iloc[split:, :]


# In[11]:


y_train, y_valid = y.iloc[:split], y.iloc[split:]


# In[12]:


get_ipython().run_cell_magic('time', '', 'xgb = XGBClassifier()\nxgb.fit(X_train, y_train)')


# In[ ]:


xgb_valid_pred = xgb.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, xgb_valid_pred)

