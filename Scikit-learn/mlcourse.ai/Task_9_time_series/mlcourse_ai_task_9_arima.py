#!/usr/bin/env python
# coding: utf-8

# # <center>Тема 9. Анализ временных рядов в Python</center>
# ## <center>Часть 2. Смерти из-за несчастного случая в США</center>

# Известно ежемесячное число смертей в результате случайного случая в США с января 1973 по декабрь 1978, необходимо построить прогноз на следующие 2 года.

# In[1]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 12, 10
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import product

def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))


# In[2]:


deaths = pd.read_csv('../../data/accidental-deaths-in-usa-monthly.csv',
                   index_col=['Month'], parse_dates=['Month'])
deaths.rename(columns={'Accidental deaths in USA: monthly, 1973 ? 1978': 'num_deaths'}, inplace=True)
deaths['num_deaths'].plot()
plt.ylabel('Accidental deaths');


# Проверка стационарности и STL-декомпозиция ряда:

# In[3]:


sm.tsa.seasonal_decompose(deaths['num_deaths']).plot()
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(deaths['num_deaths'])[1])


# ### Стационарность

# Критерий Дики-Фуллера не отвергает гипотезу нестационарности, но небольшой тренд остался. Попробуем сезонное дифференцирование; сделаем на продифференцированном ряде STL-декомпозицию и проверим стационарность:

# In[4]:


deaths['num_deaths_diff'] = deaths['num_deaths'] - deaths['num_deaths'].shift(12)
sm.tsa.seasonal_decompose(deaths['num_deaths_diff'][12:]).plot()
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(deaths['num_deaths_diff'][12:])[1])


# Критерий Дики-Фуллера отвергает гипотезу нестационарности, но полностью избавиться от тренда не удалось. Попробуем добавить ещё обычное дифференцирование:

# In[5]:


deaths['num_deaths_diff2'] = deaths['num_deaths_diff'] - deaths['num_deaths_diff'].shift(1)
sm.tsa.seasonal_decompose(deaths['num_deaths_diff2'][13:]).plot()
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(deaths['num_deaths_diff2'][13:])[1])


# Гипотеза нестационарности уверенно отвергается, и визуально ряд выглядит лучше — тренда больше нет. 

# ## Подбор модели

# Посмотрим на ACF и PACF полученного ряда:

# In[6]:


ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(deaths['num_deaths_diff2'][13:].values.squeeze(), lags=58, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(deaths['num_deaths_diff2'][13:].values.squeeze(), lags=58, ax=ax);


# Начальные приближения: Q=2, q=1, P=2, p=2

# In[7]:


ps = range(0, 3)
d=1
qs = range(0, 1)
Ps = range(0, 3)
D=1
Qs = range(0, 3)


# In[8]:


parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[9]:


get_ipython().run_cell_magic('time', '', 'results = []\nbest_aic = float("inf")\n\n\n\nfor param in parameters_list:\n    #try except нужен, потому что на некоторых наборах параметров модель не обучается\n    try:\n        model=sm.tsa.statespace.SARIMAX(deaths[\'num_deaths\'], order=(param[0], d, param[1]), \n                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)\n    #выводим параметры, на которых модель не обучается и переходим к следующему набору\n    except ValueError:\n        print(\'wrong parameters:\', param)\n        continue\n    aic = model.aic\n    #сохраняем лучшую модель, aic, параметры\n    if aic < best_aic:\n        best_model = model\n        best_aic = aic\n        best_param = param\n    results.append([param, model.aic])\n    \nwarnings.filterwarnings(\'default\')')


# Если в предыдущей ячейке возникает ошибка, убедитесь, что обновили statsmodels до версии не меньше 0.8.0rc1.

# In[10]:


result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())


# Лучшая модель:

# In[11]:


print(best_model.summary())


# Её остатки:

# In[12]:


plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Критерий Стьюдента: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])


# Остатки несмещены (подтверждается критерием Стьюдента), стационарны (подтверждается критерием Дики-Фуллера и визуально), неавтокоррелированы (подтверждается критерием Льюнга-Бокса и коррелограммой).
# Посмотрим, насколько хорошо модель описывает данные:

# In[13]:


deaths['model'] = best_model.fittedvalues
deaths['num_deaths'].plot()
deaths['model'][13:].plot(color='r')
plt.ylabel('Accidental deaths');


# ### Прогноз

# In[14]:


from dateutil.relativedelta import relativedelta
deaths2 = deaths[['num_deaths']]
date_list = [pd.datetime.strptime("1979-01-01", "%Y-%m-%d") + relativedelta(months=x) for x in range(0,24)]
future = pd.DataFrame(index=date_list, columns=deaths2.columns)
deaths2 = pd.concat([deaths2, future])
deaths2['forecast'] = best_model.predict(start=72, end=100)

deaths2['num_deaths'].plot(color='b')
deaths2['forecast'].plot(color='r')
plt.ylabel('Accidental deaths');

