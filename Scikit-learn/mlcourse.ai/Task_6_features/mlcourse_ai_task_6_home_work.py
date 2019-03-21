#!/usr/bin/env python
# coding: utf-8

# # <center>Домашнее задание 6 (демо). Линейная регрессия, Lasso и RF-регрессия в задаче по определению качества вина</center>
# 
# 
# **Заполните пропущенный код и ответьте на вопросы в [онлайн-форме](https://docs.google.com/forms/d/1gsNxgkd0VqidZp4lh9mnCQnJw3b0IFR1C4WBES86J40).**

# In[6]:


import numpy as np
import pandas as pd
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor


# **Будем работать с набором данных по качеству белого вина (репозиторий UCI).**
# **Загружаем данные.**

# In[4]:


data = pd.read_csv('winequality-white.csv', sep=',')


# In[5]:


data.head()


# In[7]:


data.info()


# **Отделите целевой признак, разделите обучающую выборку в отношении 7:3 (30% - под оставленную выборку, пусть random_state=17) и отмасштабируйте данные с помощью StandardScaler.**

# In[11]:


y = data['quality']
X = data.drop(['quality'], axis=1)

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)


# ## Линейная регрессия

# **Обучите простую линейную регрессию.**

# In[17]:


linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)


# **<font color='red'>Вопрос 1:</font> Каковы среднеквадратичные ошибки линейной регрессии на обучающей и отложенной выборках?**

# In[20]:


print("Mean squared error (train): %.3f" % mean_squared_error(y_train, linreg.predict(X_train_scaled)))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, linreg.predict(X_holdout_scaled)))


# **Посмотрите на коэффициенты модели и отранжируйте признаки по влиянию на качество вина (учтите, что большие по модулю отрицательные значения коэффициентов тоже говорят о сильном влиянии). Создайте для этого новый небольшой DataFrame.**<br>
# **<font color='red'>Вопрос 2:</font> Какой признак линейная регрессия считает наиболее сильно влияющим на качество вина?**

# In[24]:


linreg_coef = pd.DataFrame(linreg.coef_, data.columns[: -1], columns=['coef'])
linreg_coef.sort_values(by='coef', ascending=False)


# ## Lasso-регрессия

# **Обучите Lasso-регрессию с небольшим коэффициентом $\alpha = 0.01$ (слабая регуляризация). Пусть опять random_state=17.**

# In[25]:


lasso1 = Lasso(alpha=0.001, random_state=17)
lasso1.fit(X_train_scaled, y_train)


# **Посмотрите на коэффициенты модели и отранжируйте признаки по влиянию на качество вина. Какой признак "отвалился" первым, то есть наименее важен для объяснения целевого признака в модели Lasso?**

# In[27]:


lasso1_coef = pd.DataFrame(lasso1.coef_, data.columns[: -1], columns=['coef'])
lasso1_coef.sort_values(by='coef', ascending=False)


# **Теперь определите лучшее значение $\alpha$ в процессе кросс-валидации 5-кратной кросс-валидации. Используйте LassoCV и random_state=17.**

# In[28]:


alphas = np.logspace(-6, 2, 200)
lasso_cv = LassoCV(alphas=alphas, random_state=17, cv=5)
lasso_cv.fit(X_train_scaled, y_train)


# In[29]:


lasso_cv.alpha_


# **Выведите коэффициенты "лучшего" Lasso в порядке убывания влияния на качество вина. **<br>
# **<font color='red'>Вопрос 3:</font> Какой признак "обнулился первым" в настроенной модели LASSO?**

# In[31]:


lasso_cv_coef = pd.DataFrame(lasso_cv.coef_, data.columns[: -1], columns=['coef'])
lasso_cv_coef.sort_values(by='coef', ascending=False)


# **Оцените среднеквадратичную ошибку модели на обучающей и тестовой выборках.**<br>
# **<font color='red'>Вопрос 4:</font> Каковы среднеквадратичные ошибки настроенной LASSO-регрессии на обучающей и отложенной выборках?**

# In[32]:


print("Mean squared error (train): %.3f" % mean_squared_error(y_train, lasso_cv.predict(X_train_scaled)))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, lasso_cv.predict(X_holdout_scaled)))


# ## Случайный лес

# **Обучите случайный лес с параметрами "из коробки", фиксируя только random_state=17.**

# In[40]:


forest = RandomForestRegressor(random_state=17)
forest.fit(X_train, y_train)


# **<font color='red'>Вопрос 5:</font> Каковы среднеквадратичные ошибки случайного леса на обучающей выборке, на кросс-валидации (cross_val_score с scoring='neg_mean_squared_error' и остальными параметрами по умолчанию) и на отложенной выборке?**

# In[41]:


print("Mean squared error (train): %.3f" % mean_squared_error(y_train, forest.predict(X_train)))
print("Mean squared error (cv): %.3f" % np.mean(cross_val_score(forest, X_train, y_train, 
                                                                cv=5, scoring='neg_mean_squared_error')))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, forest.predict(X_holdout)))


# **Настройте параметры min_samples_leaf и max_depth с помощью GridSearchCV и опять проверьте качество модели на кросс-валидации и на отложенной выборке.**

# In[42]:


forest_params = {'max_depth': list(range(10, 25)), 
                 'min_samples_leaf': list(range(1, 8)),
                 'max_features': list(range(6,12))}

locally_best_forest = GridSearchCV(forest, forest_params, cv=5)
locally_best_forest.fit(X_train, y_train)


# In[43]:


locally_best_forest.best_params_, locally_best_forest.best_score_


# In[49]:


forest_v2 = RandomForestRegressor(max_depth=19, max_features=7, min_samples_leaf=1, random_state=17)
forest_v2.fit(X_train, y_train)


# **К сожалению, результаты  GridSearchCV не полностью воспроизводимы (могут отличаться на разных платформах даже при фиксировании *random_state*). Поэтому обучите лес с параметрами max_depth=19, max_features=7, и min_samples_leaf=1 (лучшие в моем случае).**<br>
# **<font color='red'>Вопрос 6:</font> Каковы среднеквадратичные ошибки настроенного случайного леса на обучающей выборке, на кросс-валидации (cross_val_score с scoring='neg_mean_squared_error') и на отложенной выборке?**

# In[50]:


err = np.mean(cross_val_score(forest_v2, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))


# In[51]:


print("Mean squared error (cv): %.3f" % err)
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, forest_v2.predict(X_holdout)))


# **Оцените важность признаков с помощью случайного леса.**<br>
# **<font color='red'>Вопрос 7:</font> Какой признак оказался главным в настроенной модели случайного леса?**

# In[53]:


rf_importance = pd.DataFrame(forest_v2.feature_importances_, data.columns[: -1], columns=['importance'])
rf_importance.sort_values(by='importance', ascending=False)


# **Сделайте выводы о качестве моделей и оценках влияния признаков на качество вина с помощью этих трех моделей.**
