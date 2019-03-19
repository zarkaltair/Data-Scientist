#!/usr/bin/env python
# coding: utf-8

# # <center>Тема 4. Линейные модели классификации и регрессии
# ## <center>  Практика. Идентификация пользователя с помощью логистической регрессии
# 
# Тут мы воспроизведем парочку бенчмарков нашего соревнования и вдохновимся побить третий бенчмарк, а также остальных участников. Веб-формы для отправки ответов тут не будет, ориентир – [leaderboard](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/leaderboard) соревнования.

# In[1]:


import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns


# ### 1. Загрузка и преобразование данных
# Зарегистрируйтесь на [Kaggle](www.kaggle.com), если вы не сделали этого раньше, зайдите на [страницу](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) соревнования и скачайте данные. Первым делом загрузим обучающую и тестовую выборки и посмотрим на данные.

# In[2]:


# загрузим обучающую и тестовую выборки
train_df = pd.read_csv('train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('test_sessions.csv',
                      index_col='session_id')

# приведем колонки time1, ..., time10 к временному формату
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# отсортируем данные по времени
train_df = train_df.sort_values(by='time1')

# посмотрим на заголовок обучающей выборки
train_df.head()


# В обучающей выборке содержатся следующие признаки:
#     - site1 – индекс первого посещенного сайта в сессии
#     - time1 – время посещения первого сайта в сессии
#     - ...
#     - site10 – индекс 10-го посещенного сайта в сессии
#     - time10 – время посещения 10-го сайта в сессии
#     - target – целевая переменная, 1 для сессий Элис, 0 для сессий других пользователей
#     
# Сессии пользователей выделены таким образом, что они не могут быть длиннее получаса или 10 сайтов. То есть сессия считается оконченной либо когда пользователь посетил 10 сайтов подряд либо когда сессия заняла по времени более 30 минут.
# 
# В таблице встречаются пропущенные значения, это значит, что сессия состоит менее, чем из 10 сайтов. Заменим пропущенные значения нулями и приведем признаки к целому типу. Также загрузим словарь сайтов и посмотрим, как он выглядит:

# In[3]:


# приведем колонки site1, ..., site10 к целочисленному формату и заменим пропуски нулями
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# загрузим словарик сайтов
with open(r"site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# датафрейм словарика сайтов
sites_dict_df = pd.DataFrame(list(site_dict.keys()), 
                             index=list(site_dict.values()), 
                             columns=['site'])
print(u'всего сайтов:', sites_dict_df.shape[0])
sites_dict_df.head()


# Выделим целевую переменную и объединим выборки, чтобы вместе привести их к разреженному формату.

# In[4]:


# наша целевая переменная
y_train = train_df['target']

# объединенная таблица исходных данных
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# индекс, по которому будем отделять обучающую выборку от тестовой
idx_split = train_df.shape[0]


# Для самой первой модели будем использовать только посещенные сайты в сессии (но не будем обращать внимание на временные признаки). За таким выбором данных для модели стоит такая идея:  *у Элис есть свои излюбленные сайты, и чем чаще вы видим эти сайты в сессии, тем выше вероятность, что это сессия Элис и наоборот.*
# 
# Подготовим данные, из всей таблицы выберем только признаки `site1, site2, ... , site10`. Напомним, что пропущенные значения заменены нулем. Вот как выглядят первые строки таблицы:

# In[5]:


# табличка с индексами посещенных сайтов в сессии
full_sites = full_df[sites]
full_sites.head()


# Сессии представляют собой последовательность индексов сайтов и данные в таком виде неудобны для линейных методов. В соответствии с нашей гипотезой (у Элис есть излюбленные сайты) надо преобразовать эту таблицу таким образом, чтобы каждому возможному сайту соответствовал свой отдельный признак (колонка), а его значение равнялось бы количеству посещений этого сайта в сессии. Это делается в две строчки:

# In[6]:


from scipy.sparse import csr_matrix


# In[15]:


get_ipython().run_line_magic('pinfo', 'csr_matrix')


# In[7]:


# последовательность с индексами
sites_flatten = full_sites.values.flatten()

# искомая матрица
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]


# In[8]:


full_sites_sparse.shape


# In[9]:


X_train_sparse = full_sites_sparse[: idx_split]
X_test_sparse = full_sites_sparse[idx_split:]


# In[10]:


X_train_sparse.shape, y_train.shape


# Еще один плюс использования разреженных матриц в том, что для них имеются специальные реализации как матричных операций, так и алгоритмов машинного обучения, что подчас позволяет ощутимо ускорить операции за счет особенностей структуры данных. Это касается и логистической регрессии. Вот теперь у нас все готово для построения нашей первой модели.
# 
# ### 2. Построение первой модели
# 
# Итак, у нас есть алгоритм и данные для него, построим нашу первую модель, воспользовавшись релизацией [логистической регрессии](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) из пакета `sklearn` с параметрами по умолчанию. Первые 90% данных будем использовать для обучения (обучающая выборка отсортирована по времени), а оставшиеся 10% для проверки качества (validation). 
# 
# **Напишите простую функцию, которая будет возвращать качество модели на отложенной выборке, и обучите наш первый классификатор**.

# In[11]:


def get_auc_lr_valid(X, y, C=1.0, ratio = 0.9, seed=17):
    '''
    X, y – выборка
    ratio – в каком отношении поделить выборку
    C, seed – коэф-т регуляризации и random_state 
              логистической регрессии
    '''
    train_len = int(ratio * X.shape[0])
    X_train = X[:train_len, :]
    X_test = X[train_len:, :]
    y_train = y[:train_len]
    y_test = y[train_len:]
    
    logit = LogisticRegression(C=C, n_jobs=-1, random_state=seed, solver='lbfgs')
    logit.fit(X_train, y_train)
    
    valid_pred = logit.predict_proba(X_test)[:, 1]
    
    return roc_auc_score(y_test, valid_pred)


# **Посмотрите, какой получился ROC AUC на отложенной выборке.**

# In[29]:


get_auc_lr_valid(X_train_sparse, y_train)


# Будем считать эту модель нашей первой отправной точкой (baseline). Для построения модели для прогноза на тестовой выборке **необходимо обучить модель заново уже на всей обучающей выборке** (пока наша модель обучалась лишь на части данных), что повысит ее обобщающую способность:

# In[12]:


# функция для записи прогнозов в файл
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# **Обучите модель на всей выборке, сделайте прогноз для тестовой выборки и сделайте посылку в соревновании**.

# In[13]:


logit = LogisticRegression(n_jobs=-1, random_state=17, solver='lbfgs')
logit.fit(X_train_sparse, y_train)


# In[14]:


test_pred = logit.predict_proba(X_test_sparse)[:, 1]


# In[15]:


test_pred.shape


# In[16]:


pd.Series(test_pred, 
          index=range(1, test_pred.shape[0] + 1), 
          name='target').to_csv('benchmark_v1.csv', header=True, index_label='session_id')


# In[17]:


get_ipython().system('head benchmark_v1.csv')


# Если вы выполните эти действия и загрузите ответ на [странице](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) соревнования, то воспроизведете первый бенчмарк "Logit".
# 
# ### 3. Улучшение модели, построение новых признаков

# Создайте такой признак, который будет представлять собой число вида ГГГГММ от той даты, когда проходила сессия, например 201407 -- 2014 год и 7 месяц. Таким образом, мы будем учитывать помесячный [линейный тренд](http://people.duke.edu/~rnau/411trend.htm) за весь период предоставленных данных.

# In[18]:


train_df[times].head()


# In[19]:


new_feat_train = pd.DataFrame(index=train_df.index)
new_feat_test = pd.DataFrame(index=test_df.index)


# In[20]:


new_feat_train['year_month'] = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)
new_feat_test['year_month'] = test_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)


# In[21]:


train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).head()


# In[22]:


new_feat_train.head()


# In[23]:


scaler = StandardScaler()
scaler.fit(new_feat_train['year_month'].values.reshape(-1, 1))

new_feat_train['year_month_scaled'] = scaler.transform(new_feat_train['year_month'].values.reshape(-1, 1))
new_feat_test['year_month_scaled'] = scaler.transform(new_feat_test['year_month'].values.reshape(-1, 1))


# In[24]:


new_feat_train.head()


# Добавьте новый признак, предварительно отмасштабировав его с помощью `StandardScaler`, и снова посчитайте ROC AUC на отложенной выборке.

# In[26]:


X_train_sparse.shape, X_train_sparse_new.shape


# In[39]:


X_train_sparse_new = csr_matrix(hstack([X_train_sparse, 
                             new_feat_train['year_month_scaled'].values.reshape(-1, 1)]))
X_test_sparse_new = csr_matrix(hstack([X_test_sparse, 
                             new_feat_test['year_month_scaled'].values.reshape(-1, 1)]))


# In[27]:


get_ipython().run_cell_magic('time', '', 'get_auc_lr_valid(X_train_sparse_new, y_train)')


# **Добавьте два новых признака: start_hour и morning.**
# 
# Признак `start_hour` – это час в который началась сессия (от 0 до 23), а бинарный признак `morning` равен 1, если сессия началась утром и 0, если сессия началась позже (будем считать, что утро это если `start_hour равен` 11 или меньше).
# 
# **Посчитйте ROC AUC на отложенной выборке для выборки с:**
# - сайтами, `start_month` и `start_hour`
# - сайтами, `start_month` и `morning`
# - сайтами, `start_month`, `start_hour` и `morning`

# In[28]:


train_df[times].head(3)


# In[31]:


new_feat_train['start_hour'] = train_df['time1'].apply(lambda ts: ts.hour)
new_feat_test['start_hour'] = test_df['time1'].apply(lambda ts: ts.hour)


# In[32]:


new_feat_train['start_hour'].head(3)


# In[34]:


new_feat_train.head()


# In[36]:


scaler.fit(new_feat_train['start_hour'].values.reshape(-1, 1))

new_feat_train['start_hour_scaled'] = scaler.transform(new_feat_train['start_hour'].values.reshape(-1, 1))
new_feat_test['start_hour_scaled'] = scaler.transform(new_feat_test['start_hour'].values.reshape(-1, 1))


# In[40]:


X_train_sparse_new_hour = csr_matrix(hstack([X_train_sparse_new, 
                             new_feat_train['start_hour_scaled'].values.reshape(-1, 1)]))
X_test_sparse_new_hour = csr_matrix(hstack([X_test_sparse_new, 
                             new_feat_test['start_hour_scaled'].values.reshape(-1, 1)]))


# In[38]:


get_ipython().run_cell_magic('time', '', 'get_auc_lr_valid(X_train_sparse_new_hour, y_train)')


# In[41]:


logit = LogisticRegression(n_jobs=-1, random_state=17, solver='lbfgs')
logit.fit(X_train_sparse_new_hour, y_train)
test_pred = logit.predict_proba(X_test_sparse_new_hour)[:, 1]


# In[42]:


pd.Series(test_pred, 
          index=range(1, test_pred.shape[0] + 1), 
          name='target').to_csv('benchmark_v2.csv', header=True, index_label='session_id')


# In[47]:


new_feat_train['morning'] = train_df['time1'].apply(lambda ts: int(ts.hour <= 11))
new_feat_test['morning'] = test_df['time1'].apply(lambda ts: int(ts.hour <= 11))


# In[48]:


X_train_sparse_new_hour_morning = csr_matrix(hstack([X_train_sparse_new_hour, 
                             new_feat_train['morning'].values.reshape(-1, 1)]))
X_test_sparse_new_hour_morning = csr_matrix(hstack([X_test_sparse_new_hour, 
                             new_feat_test['morning'].values.reshape(-1, 1)]))


# In[50]:


get_ipython().run_cell_magic('time', '', 'get_auc_lr_valid(X_train_sparse_new_hour_morning, y_train)')


# In[51]:


logit = LogisticRegression(n_jobs=-1, random_state=17, solver='lbfgs')
logit.fit(X_train_sparse_new_hour_morning, y_train)
test_pred = logit.predict_proba(X_test_sparse_new_hour_morning)[:, 1]


# In[52]:


pd.Series(test_pred, 
          index=range(1, test_pred.shape[0] + 1), 
          name='target').to_csv('benchmark_v3.csv', header=True, index_label='session_id')


# ### 4. Подбор коэффицициента регуляризации
# 
# Итак, мы ввели признаки, которые улучшают качество нашей модели по сравнению с первым бейслайном. Можем ли мы добиться большего значения метрики? После того, как мы сформировали обучающую и тестовую выборки, почти всегда имеет смысл подобрать оптимальные гиперпараметры -- характеристики модели, которые не изменяются во время обучения. Например, на 3 неделе вы проходили решающие деревья, глубина дерева это гиперпараметр, а признак, по которому происходит ветвление и его значение -- нет. В используемой нами логистической регрессии веса каждого признака изменяются и во время обучения находится их оптимальные значения, а коэффициент регуляризации остается постоянным. Это тот гиперпараметр, который мы сейчас будем оптимизировать.
# 
# Посчитайте качество на отложенной выборке с коэффициентом регуляризации, который по умолчанию `C=1`:

# In[ ]:


# Ваш код здесь


# Постараемся побить этот результат за счет оптимизации коэффициента регуляризации. Возьмем набор возможных значений C и для каждого из них посчитаем значение метрики на отложенной выборке.
# 
# Найдите `C` из `np.logspace(-3, 1, 10)`, при котором ROC AUC на отложенной выборке максимален. 

# In[ ]:


# Ваш код здесь


# Наконец, обучите модель с найденным оптимальным значением коэффициента регуляризации и с построенными признаками `start_hour`, `start_month` и `morning`. Если вы все сделали правильно и загрузите это решение, то повторите второй бенчмарк соревнования.

# In[ ]:


# Ваш код здесь

