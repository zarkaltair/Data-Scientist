#!/usr/bin/env python
# coding: utf-8

# # <center> Домашнее задание № 9. (демо). Решение
# ## <center> Анализ временных рядов
#     
# **Заполните пропущенный код и ответьте на вопросы в [онлайн-форме](https://docs.google.com/forms/d/1ijk4aFKY5plPiI8z3Mgi3i1Ln94VBY9SSt6xGIdVVFQ/).**

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go
import requests
import pandas as pd

print(__version__) # need 1.9.0 or greater

init_notebook_mode(connected = True)


def plotly_df(df, title = ''):
    data = []
    
    for column in df.columns:
        trace = go.Scatter(
            x = df.index,
            y = df[column],
            mode = 'lines',
            name = column
        )
        data.append(trace)
    
    layout = dict(title = title)
    fig = dict(data = data, layout = layout)
    iplot(fig, show_link=False)


# ## Подготавливаем данные

# In[2]:


df = pd.read_csv('../../data/wiki_machine_learning.csv', sep = ' ')
df = df[df['count'] != 0]
df.head()


# In[3]:


df.shape


# ## Предсказываем с помощью FB Prophet
# Будем обучаться на первых 5 месяцах и предсказывать число поездок за июнь.

# In[4]:


df.date = pd.to_datetime(df.date)


# In[5]:


plotly_df(df.set_index('date')[['count']])


# In[6]:


from fbprophet import Prophet


# In[7]:


predictions = 30

df = df[['date', 'count']]
df.columns = ['ds', 'y']
df.tail()


# In[8]:


train_df = df[:-predictions].copy()


# In[9]:


m = Prophet()
m.fit(train_df);


# In[10]:


future = m.make_future_dataframe(periods=predictions)
future.tail()


# In[11]:


forecast = m.predict(future)
forecast.tail()


# **Вопрос 1:** Какое предсказание числа просмотров wiki-страницы на 20 января? Ответ округлите до целого числа.
# 
# **Ответ:** 3833

# In[12]:


m.plot(forecast)


# In[13]:


m.plot_components(forecast)


# In[14]:


cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 
                                   'yhat_upper']].join(df.set_index('ds'))


# In[15]:


import numpy as np
cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']
print('MAPE = ', round(np.mean(abs(cmp_df[-predictions:]['p'])), 2))
print('MAE = ', round(np.mean(abs(cmp_df[-predictions:]['e'])), 2))


# Оценим качество предсказания по последним 30 точкам.
# 
# **Вопрос 2**: Какое получилось MAPE?
# 
# **Ответ:** 38.38
# 
# **Вопрос 3**: Какое получилось MAE?
# 
# **Ответ:** 712.86

# ## Предсказываем с помощью ARIMA

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (15, 10)


# **Вопрос 4:** Проверим стационарность ряда с помощью критерия Дики-Фулера. Является ли ряд стационарным? Какое значение p-value?

# In[17]:


sm.tsa.seasonal_decompose(train_df['y'].values, freq=7).plot();
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(train_df['y'])[1])


# А вот сезонно продифференцированный ряд уже будет стационарным.

# In[18]:


train_df.set_index('ds', inplace=True)


# In[19]:


train_df['y_diff'] = train_df.y - train_df.y.shift(7)
sm.tsa.seasonal_decompose(train_df.y_diff[7:].values, freq=7).plot();
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(train_df.y_diff[8:])[1])


# In[20]:


ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(train_df.y_diff[13:].values.squeeze(), lags=48, ax=ax)

ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(train_df.y_diff[13:].values.squeeze(), lags=48, ax=ax)


# Начальные приближения:
# * Q = 1
# * q = 3
# * P = 3
# * p = 1

# In[21]:


ps = range(0, 2)
ds = range(0, 2)
qs = range(0, 4)
Ps = range(0, 4)
Ds = range(0, 3)
Qs = range(0, 2)


# In[22]:


from itertools import product

parameters = product(ps, ds, qs, Ps, Ds, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[23]:


get_ipython().run_cell_magic('time', '', 'import warnings\nfrom tqdm import tqdm\nresults1 = []\nbest_aic = float("inf")\nwarnings.filterwarnings(\'ignore\')\n\nfor param in tqdm(parameters_list):\n    #try except нужен, потому что на некоторых наборах параметров модель не обучается\n    try:\n        model=sm.tsa.statespace.SARIMAX(train_df[\'y\'], order=(param[0], param[1], param[2]), \n                                        seasonal_order=(param[3], param[4], param[5], 7)).fit(disp=-1)\n    #выводим параметры, на которых модель не обучается и переходим к следующему набору\n    except (ValueError, np.linalg.LinAlgError):\n        continue\n    aic = model.aic\n    #сохраняем лучшую модель, aic, параметры\n    if aic < best_aic:\n        best_model = model\n        best_aic = aic\n        best_param = param\n    results1.append([param, model.aic])')


# In[24]:


result_table1 = pd.DataFrame(results1)
result_table1.columns = ['parameters', 'aic']
print(result_table1.sort_values(by = 'aic', ascending=True).head())


# Если рассматривать предложенные в форме варианты:

# In[25]:


result_table1[result_table1['parameters'].isin([(1, 0, 2, 3, 1, 0),
                                                (1, 1, 2, 3, 2, 1),
                                                (1, 1, 2, 3, 1, 1),
                                                (1, 0, 2, 3, 0, 0)])]


# Теперь то же самое, но для ряда с преобразованием Бокса-Кокса.

# In[26]:


import scipy.stats
train_df['y_box'], lmbda = scipy.stats.boxcox(train_df['y']) 
print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)


# In[27]:


results2 = []
best_aic = float("inf")

for param in tqdm(parameters_list):
    #try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
        model=sm.tsa.statespace.SARIMAX(train_df['y_box'], order=(param[0], param[1], param[2]), 
                                        seasonal_order=(param[3], param[4], param[5], 7)).fit(disp=-1)
    #выводим параметры, на которых модель не обучается и переходим к следующему набору
    except (ValueError, np.linalg.LinAlgError):
        continue
    aic = model.aic
    #сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results2.append([param, model.aic])
    
warnings.filterwarnings('default')


# In[28]:


result_table2 = pd.DataFrame(results2)
result_table2.columns = ['parameters', 'aic']
print(result_table2.sort_values(by = 'aic', ascending=True).head())


# Если рассматривать предложенные в форме варианты:

# In[29]:


result_table2[result_table2['parameters'].isin([(1, 0, 2, 3, 1, 0),
                                                (1, 1, 2, 3, 2, 1),
                                                (1, 1, 2, 3, 1, 1),
                                                (1, 0, 2, 3, 0, 0)])].sort_values(by = 'aic')


# **Вопрос 5**: Модель SARIMAX c какими параметрами лучшая по AIC-критерию?

# **Ответ:** из предложенных вариантов подходят первый (D = 1, d = 0, Q = 0, q = 2, P = 3, p = 1) и второй (D = 2, d = 1, Q = 1, q = 2, P = 3, p = 1) в зависимости от того, делать ли вначале преобразование Бокса-Кокса или нет.

# Посмотрим на прогноз лучшей по AIC модели.

# In[30]:


print(best_model.summary())


# In[31]:


plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Критерий Стьюдента: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])


# In[32]:


def invboxcox(y,lmbda):
    # обратное преобразование Бокса-Кокса
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda * y + 1) / lmbda))


# In[33]:


train_df['arima_model'] = invboxcox(best_model.fittedvalues, lmbda)

train_df.y.tail(200).plot()
train_df.arima_model[13:].tail(200).plot(color='r')
plt.ylabel('wiki pageviews');

