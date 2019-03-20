#!/usr/bin/env python
# coding: utf-8

# # <center> Домашнее задание № 5 (Демо).
# ## <center> Логистическая регрессия и случайный лес в задаче кредитного скоринга

# [Веб-форма](https://docs.google.com/forms/d/1HASy2b_FLBHBCzzpG-TbnbB6gqhB-qwznQxU2vaoSgc/) для ответов.
# 
# #### Нашей главной задачей будет построение модели для задачи кредитного скоринга.
# 
# Но для разминки решите первое задание :)
# 
# **Задание 1.** В зале суда есть 5 присяжных, каждый из них по отдельности с вероятностью 70% может правильно определить, виновен подсудимый или нет. С какой вероятностью они все вместе вынесут правильный вердикт, если решение принимается большинством голосов?
# - 70.00%
# - 83.20%
# - 83.70%
# - 87.50%
# 
# Теперь перейдем непосредственно к машинному обучению.
# 
# #### Данные представлены следующим образом:
# 
# ##### Прогнозируемая  переменная
# * SeriousDlqin2yrs	      – Человек не выплатил данный кредит в течение 90 дней; возможные значения  1/0 
# 
# ##### Независимые признаки
# * age	                          –  Возраст заёмщика кредитных средств; тип - integer
# * NumberOfTime30-59DaysPastDueNotWorse	 – Количество раз, когда человек имел просрочку выплаты других кредитов более 30-59 дней, но не больше в течение последних двух лет; тип -	integer
# * DebtRatio  – 	Ежемесячный отчисления на задолжености(кредиты,алименты и т.д.)  / совокупный месячный доход 	percentage; тип -	real
# * MonthlyIncome	 – Месячный доход в долларах; тип -	real
# * NumberOfTimes90DaysLate  – Количество раз, когда человек имел просрочку выплаты других кредитов более 90 дней; тип -	integer
# * NumberOfTime60-89DaysPastDueNotWorse – 	Количество раз, когда человек имел просрочку выплаты других кредитов более 60-89 дней, но не больше в течение последних двух лет; тип -	integer
# * NumberOfDependents  – Число человек в семье кредитозаёмщика; тип -	integer

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
# отключим предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np


# In[2]:


## Сделаем функцию, которая будет заменять NaN значения на медиану в каждом столбце таблицы 
def delete_nan(table):
    for col in table.columns:
        table[col] = table[col].fillna(table[col].median())
    return table   


# In[3]:


## Считываем данные
data = pd.read_csv('credit_scoring_sample.csv', sep =',')
data.head()


# In[4]:


## Рассмотрим типы считанных данных
data.dtypes


# In[5]:


## Посмотрим на распределение классов в зависимой переменной

ax = data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')
ax.set_xlabel("number_of_observations")
ax.set_ylabel("unique_value")
ax.set_title("Target distribution")

print('Distribution of target')
data['SeriousDlqin2yrs'].value_counts()/data.shape[0]


# In[6]:


## Выберем названия всех признаков из таблицы, кроме прогнозируемого

independent_columns_names = data.columns.values
independent_columns_names = [x for x in data if x != 'SeriousDlqin2yrs']
independent_columns_names


# In[7]:


## Применяем функцию, заменяющую все NaN значения на медианное значение соответствующего столбца
table = delete_nan(data)


# In[8]:


## Разделяем таргет и признаки 
X = table[independent_columns_names]
y = table['SeriousDlqin2yrs']


# # Бутстрэп

# **Задание 2.** Сделайте интервальную оценку среднего возраста (age) для клиентов, которые просрочили выплату кредита, с 90% "уверенностью". Используйте пример из статьи, поставьте `np.random.seed(0)`, как это сделано в статье.

# In[25]:


def get_bootstrap_samples(data, n_samples):
    # функция для генерации подвыборок с помощью бутстрэпа
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    # функция для интервальной оценки
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

# сохранение в отдельные numpy массивы данных по просрочке
churn = data[data['SeriousDlqin2yrs'] == 1]['age'].values

# ставим seed для воспроизводимости результатов
np.random.seed(0)

# генерируем выборки с помощью бутстрэра и сразу считаем по каждой из них среднее
churn_mean_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(churn, 1000)]

#  выводим интервальную оценку среднего
print("Mean interval",  stat_intervals(churn_mean_scores, 0.1))


# ## Подбор параметров для модели логистической регрессии 

# #### Одной из важных метрик качества модели является значение площади под ROC-кривой. Значение ROC-AUC лежит от 0  до 1.   Чем ближе начение метрики ROC-AUC к 1, тем качественнее происходит классификация моделью.

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

## Используем модуль LogisticRegression для построения логистической регрессии.
## Из-за несбалансированности классов в таргете добавляем параметр балансировки.
## Используем также параметр random_state=5 для воспроизводимости результатов
lr = LogisticRegression(random_state=5, class_weight= 'balanced')

## Попробуем подобрать лучший коэффициент регуляризации (коэффициент C в логистической регрессии) для модели лог.регрессии.
## Этот параметр необходим для того, чтобы подобрать оптимальную модель, которая не будет переобучена, с одной стороны, 
## и будет хорошо предсказывать значения таргета, с другой.
## Остальные параметры оставляем по умолчанию.
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}

## Для того, чтобы подобрать коэффициент регуляризации, попробуем для каждого его возможного значения посмотреть 
## значения roc-auc на стрэтифайд кросс-валидации из 5 фолдов с помощью функции StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)


# **Задание 3.**
# Сделайте GridSearch с метрикой "roc-auc" по параметру C. Какое оптимальное значение параметра С?

# In[11]:


grid_logit = GridSearchCV(lr, parameters, cv=skf, n_jobs=-1, scoring='roc_auc')
grid_logit.fit(X, y)


# In[12]:


grid_logit.best_params_


# **Задание 4.** 
# Можно ли считать лучшую модель устойчивой? (модель считаем устойчивой, если стандартное отклонение на валидации меньше 0.5%) Сохраните точность лучшей модели, она вам приходится для следующих заданий

# In[27]:


grid_logit.cv_results_['std_test_score'][1]


# In[16]:


grid_logit.best_score_


# ## Определение влияния признаков

# **Задание 5.**
# Определите самый важный признак. Важность признака определяется абсолютным значением его коэффициента. Так же нужно нормализировать все признаки, что бы можно их было корректно сравнить.

# In[28]:


from sklearn.preprocessing import StandardScaler
lr = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')
scal = StandardScaler()
lr.fit(scal.fit_transform(X), y)

pd.DataFrame({'feat': independent_columns_names,
              'coef': lr.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)


# **Задание 6.** Посчитайте долю влияния DebtRatio на предсказание. (Воспользуйтесь функцией [softmax](https://en.wikipedia.org/wiki/Softmax_function))

# In[30]:


(np.exp(lr.coef_[0]) / np.sum(np.exp(lr.coef_[0])))[2]


# **Задание 7.** 
# Давайте посмотрим как можно интерпретировать влияние наших признаков. Для этого заного оценим логистическую регрессию в абсолютных величинах. После этого посчитайте во сколько раз увеличатся шансы, что клиент не выплатит кредит, если увеличить возраст на 20 лет при всех остальных равных значениях признаков. (теоретический расчет можно посмотреть [здесь](https://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf))

# In[31]:


lr = LogisticRegression(C=0.001,  random_state=5, class_weight= 'balanced')
lr.fit(X, y)

pd.DataFrame({'feat': independent_columns_names,
              'coef': lr.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)


# In[32]:


np.exp(lr.coef_[0][0]*20)


# # Случайный лес

# In[14]:


from sklearn.ensemble import RandomForestClassifier

# Инициализируем случайный лес с 100 деревьями и сбалансированными классами 
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                            random_state=42, oob_score=True, 
                            class_weight='balanced')

## Будем искать лучшие параметры среди следующего набора
parameters = {'max_features': [1, 2, 4], 
              'min_samples_leaf': [3, 5, 7, 9], 
              'max_depth': [5,10,15]}

## Делаем опять же стрэтифайд k-fold валидацию. Инициализация которой должна у вас продолжать храниться в skf


# **Задание 8.** На сколько точность лучшей модели случайного леса выше точности логистической регрессии на валидации?

# In[15]:


rf_grid = GridSearchCV(rf, parameters, cv=skf, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X, y)
rf_grid.best_params_


# In[17]:


rf_grid.best_score_


# **Задание 9.** Определите какой признак имеет самое слабое влияние.

# In[34]:


pd.DataFrame({'feat': independent_columns_names,
              'coef': rf_grid.best_estimator_.feature_importances_}).sort_values(by='coef', ascending=False)


# ** Задание 10.** Какое наиболее существенное примущество логистической регрессии перед случайным лесом для нашей бизнес-задачи?
# 
# - меньше тратится времени для тренировки модели;
# - меньше параметров для перебора;
# - интепретируемость признаков;
# - линейные свойства алгоритма.

# # Бэггинг

# In[21]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 
              "base_estimator__C": [0.0001, 0.001, 0.01, 1, 10, 100]}


# **Задание 11.** Следующая задача обучить бэггинг классификатор (`random_state`=42). В качестве базовых классификаторов возьмите 100 логистических регрессий и на этот раз используйте не `GridSearchCV`, а `RandomizedSearchCV`. Так как перебирать все 54 варианта комбинаций долго, то поставьте максимальное число итераций 20 для `RandomizedSearchCV`. Также не забудьте передать параметр валидации `cv` и `random_state=1`. Какая лучшая точность получилась?

# In[23]:


bg_clf = BaggingClassifier(base_estimator=lr, n_estimators=100, random_state=42)
bg_clf_grid_random = RandomizedSearchCV(bg_clf, param_distributions=parameters, n_iter=20, cv=skf, random_state=1, n_jobs=-1)
bg_clf_grid_random.fit(X, y)


# In[24]:


bg_clf_grid_random.best_score_


# **Задача 12.** Дайте интерпретацию лучших параметров для бэггинга. Почему именно такие значения оказались лучшими?
# 
# - для бэггинга важно использовать как можно меньше признаков
# - бэггинг лучше работает на небольших выборках
# - меньше корреляция между одиночными моделями
# - чем больше признаков, тем меньше теряется информации
