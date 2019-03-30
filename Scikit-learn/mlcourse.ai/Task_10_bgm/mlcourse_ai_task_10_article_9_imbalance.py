#!/usr/bin/env python
# coding: utf-8

# ## Общие советы
# Есть несколько общих советов по работе с несбалансированными выборками:
# 
# - собрать больше данных
# - использовать метрики, нечувствительные к дисбалансу классов (F1, ROC AUC)
# - oversampling/undersampling - брать больше объектов мало представленного класса, и мало - частого класса
# - создать искусственные объекты, похожие на объекты редкого класса (например, алгоритмом SMOTE)
# 
# С XGBoost можно:
# - следить за тем, чтобы параметр `min_child_weight` был  мал, хотя по умолчанию он и так равен 1. 
# - задать ббольше веса некоторым объектам при инициализации `DMatrix`
# - контролировать отшошение числа представителей разных классов с помощью параметра `set_pos_weight`

# ## Подготовка данных

# In[1]:


import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# **Сгенерируем несбалансированную выборку для задачи классификации.**

# In[2]:


X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    n_classes=2,
    weights=[.9, .1],
    shuffle=True,
    random_state=123
)

print('There are {} positive instances.'.format(y.sum()))


# **Разбиваем на обучающую и тестовую выборки. Соблюдаем стратификацию.**

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=123)

print('Train set labels distibution: {}'.format(np.bincount(y_train)))
print('Test set labels distibution:  {}'.format(np.bincount(y_test)))


# **В начале игнорируем то, что выборка несбалансированная.**

# In[4]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# **Инициализируем параметры Xgboost - будем обучать композицию из 15 "пеньков".**

# In[5]:


params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':1
}

num_rounds = 15


# In[6]:


xgb_model = xgb.train(params, dtrain, num_rounds)
y_test_preds = (xgb_model.predict(dtest) > 0.5).astype('int')


# **Матрица ошибок.**

# In[7]:


pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)


# **Доля правильных ответов, точность и полнота.**

# In[8]:


print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))


# **Видно, что полнота низкая. то есть алгоритм плохо распознает объекты мало представленного класса. Если интересно находить как раз такие редкие объекты, то от такого алгоритма мало толку.**

# ## Задание весов вручную
# **При создании объекта `DMatrix` можно сразу явно указать, что вес положительных объектов в 5 раз больше, чем отрицательных.**

# In[9]:


weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 5

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights) # weights added
dtest = xgb.DMatrix(X_test)


# **Повторим обучение модели, как и в предыдущем случае.**

# In[10]:


xgb_model = xgb.train(params, dtrain, num_rounds)
y_test_preds = (xgb_model.predict(dtest) > 0.5).astype('int')


# In[11]:


pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)


# In[12]:


print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))


# **Видим, что вес объектов надо настраивать в зависимости от задачи.**

# ## Параметр `scale_pos_weight` в Xgboost
# **Задание весов вручную можно заменить на параметр `scale_pos_weight`**

# In[13]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# **Инициализируем параметр `scale_pos_weight` соотношением числа отрицательных и положительных объектов.**

# In[14]:


train_labels = dtrain.get_label()

ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
params['scale_pos_weight'] = ratio


# In[15]:


xgb_model = xgb.train(params, dtrain, num_rounds)
y_test_preds = (xgb_model.predict(dtest) > 0.5).astype('int')

pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)


# In[16]:


print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))


# **В этом случае значение параметра `scale_pos_weight` надо выбирать в зависимости от желаемого соотношения между точностью и полнотой.**

# ## Пример с оттоком клиентов телеком-компании

# **Загрузим данные и осуществим минимальную предобработку.**

# In[17]:


df = pd.read_csv('telecom_churn.csv')


# In[18]:


df.head()


# **Штаты просто занумеруем, а признаки International plan (наличие международного роуминга), Voice mail plan (наличие голосовой почтыы) и целевой Churn сделаем бинарными.**

# In[19]:


state_enc = LabelEncoder()
df['State'] = state_enc.fit_transform(df['State'])
df['International plan'] = (df['International plan'] == 'Yes').astype('int')
df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype('int')
df['Churn'] = (df['Churn']).astype('int')


# **Видим, что соотношение хороших и плохих клиентов примерно 6:1.**

# In[20]:


df['Churn'].value_counts()


# **Разделим данные на обучающую и тестовую выборки в отношении 7:3 с учетом соотношения классов. Инициализируем соотв. объекты DMatrix dtrain и dtest.**

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'],
                                                    test_size=0.3, stratify=df['Churn'], random_state=42)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


# In[22]:


params = {
    'objective':'binary:logistic',
    'max_depth': 4,
    'silent': 1,
    'eta': 0.3
}

num_rounds = 100


# In[23]:


xgb_model = xgb.train(params, dtrain, num_rounds)


# In[24]:


preds_prob = xgb_model.predict(dtest)
pred_labels = (preds_prob > 0.5).astype('int')


# In[25]:


pd.crosstab(
    pd.Series(dtest.get_label(), name='Actual'),
    pd.Series(pred_labels, name='Predicted'),
    margins=True
)


# In[26]:


print('Accuracy: {0:.2f}'.format(accuracy_score(dtest.get_label(), pred_labels)))
print('Precision: {0:.2f}'.format(precision_score(dtest.get_label(), pred_labels)))
print('Recall: {0:.2f}'.format(recall_score(dtest.get_label(), pred_labels)))
print('F1: {0:.2f}'.format(f1_score(dtest.get_label(), pred_labels)))


# **Теперь изменим параметр `scale_pos_weight` и проделаем то же самое.**

# In[27]:


params['scale_pos_weight'] = 10


# In[28]:


xgb_model = xgb.train(params, dtrain, num_rounds)


# In[29]:


preds_prob = xgb_model.predict(dtest)
pred_labels = (preds_prob > 0.5).astype('int')


# In[30]:


pd.crosstab(
    pd.Series(dtest.get_label(), name='Actual'),
    pd.Series(pred_labels, name='Predicted'),
    margins=True)


# In[31]:


print('Accuracy: {0:.2f}'.format(accuracy_score(dtest.get_label(), pred_labels)))
print('Precision: {0:.2f}'.format(precision_score(dtest.get_label(), pred_labels)))
print('Recall: {0:.2f}'.format(recall_score(dtest.get_label(), pred_labels)))
print('F1: {0:.2f}'.format(f1_score(dtest.get_label(), pred_labels)))


# **Видим, что таким образом мы настроили модель так, что она меньше ошибается в распознавании плохих клиентов.**
