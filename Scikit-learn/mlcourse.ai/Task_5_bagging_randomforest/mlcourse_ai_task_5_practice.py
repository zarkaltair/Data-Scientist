#!/usr/bin/env python
# coding: utf-8

# # <center> Тема 5. Композиции алгоритмов, случайный лес
# ## <center>Практика. Деревья решений и случайный лес в соревновании Kaggle Inclass по кредитному скорингу

# Тут веб-формы для ответов нет, ориентируйтесь на рейтинг [соревнования](https://inclass.kaggle.com/c/beeline-credit-scoring-competition-2), [ссылка](https://www.kaggle.com/t/115237dd8c5e4092a219a0c12bf66fc6) для участия.
# 
# Решается задача кредитного скоринга. 
# 
# Признаки клиентов банка:
# - Age - возраст (вещественный)
# - Income - месячный доход (вещественный)
# - BalanceToCreditLimit - отношение баланса на кредитной карте к лимиту по кредиту (вещественный)
# - DIR - Debt-to-income Ratio (вещественный)
# - NumLoans - число заемов и кредитных линий
# - NumRealEstateLoans - число ипотек и заемов, связанных с недвижимостью (натуральное число)
# - NumDependents - число членов семьи, которых содержит клиент, исключая самого клиента (натуральное число)
# - Num30-59Delinquencies - число просрочек выплат по кредиту от 30 до 59 дней (натуральное число)
# - Num60-89Delinquencies - число просрочек выплат по кредиту от 60 до 89 дней (натуральное число)
# - Delinquent90 - были ли просрочки выплат по кредиту более 90 дней (бинарный) - имеется только в обучающей выборке

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('matplotlib', 'inline')


# **Загружаем данные.**

# In[2]:


train_df = pd.read_csv('credit_scoring_train.csv', index_col='client_id')
test_df = pd.read_csv('credit_scoring_test.csv', index_col='client_id')


# In[3]:


y = train_df['Delinquent90']
train_df.drop('Delinquent90', axis=1, inplace=True)


# In[4]:


train_df.head()


# **Посмотрим на число пропусков в каждом признаке.**

# In[5]:


train_df.info()


# In[6]:


test_df.info()


# **Заменим пропуски медианными значениями.**

# In[8]:


train_df['NumDependents'].fillna(train_df['NumDependents'].median(), inplace=True)
train_df['Income'].fillna(train_df['Income'].median(), inplace=True)
test_df['NumDependents'].fillna(test_df['NumDependents'].median(), inplace=True)
test_df['Income'].fillna(test_df['Income'].median(), inplace=True)


# ### Дерево решений без настройки параметров

# **Обучите дерево решений максимальной глубины 3, используйте параметр random_state=17 для воспроизводимости результатов.**

# In[9]:


first_tree = DecisionTreeClassifier(max_depth=3, random_state=17)
first_tree.fit(train_df, y)


# **Сделайте прогноз для тестовой выборки.**

# In[11]:


first_tree_pred = first_tree.predict(test_df)


# **Запишем прогноз в файл.**

# In[12]:


def write_to_submission_file(predicted_labels, out_file,
                             target='Delinquent90', index_label="client_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(75000, 
                                                  predicted_labels.shape[0] + 75000),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[16]:


write_to_submission_file(first_tree_pred, 'credit_scoring_first_tree_v1.csv')


# **Если предсказывать вероятности дефолта для клиентов тестовой выборки, результат будет намного лучше.**

# In[14]:


first_tree_pred_probs = first_tree.predict_proba(test_df)[:, 1]


# In[17]:


write_to_submission_file(first_tree_pred_probs, 'credit_scoring_first_tree_v2.csv')


# ## Дерево решений с настройкой параметров с помощью GridSearch

# **Настройте параметры дерева с помощью `GridSearhCV`, посмотрите на лучшую комбинацию параметров и среднее качество на 5-кратной кросс-валидации. Используйте параметр `random_state=17` (для воспроизводимости результатов), не забывайте про распараллеливание (`n_jobs=-1`).**

# In[19]:


tree_params = {'max_depth': list(range(3, 8)), 
               'min_samples_leaf': list(range(5, 13))}

locally_best_tree = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)
locally_best_tree.fit(train_df, y)


# In[20]:


locally_best_tree.best_params_, round(locally_best_tree.best_score_, 3)


# **Сделайте прогноз для тестовой выборки и пошлите решение на Kaggle.**

# In[21]:


tuned_tree_pred_probs = locally_best_tree.predict_proba(test_df)[:, 1]


# In[26]:


write_to_submission_file(tuned_tree_pred_probs, 'credit_scoring_first_tree_v3.csv')


# ### Случайный лес без настройки параметров

# **Обучите случайный лес из деревьев неограниченной глубины, используйте параметр `random_state=17` для воспроизводимости результатов.**

# In[23]:


first_forest = RandomForestClassifier(random_state=17)
first_forest.fit(train_df, y)


# In[24]:


first_forest_pred = first_forest.predict_proba(test_df)[:, 1]


# **Сделайте прогноз для тестовой выборки и пошлите решение на Kaggle.**

# In[27]:


write_to_submission_file(first_forest_pred, 'credit_scoring_first_tree_v4.csv')


# ### Случайный лес c настройкой параметров

# **Настройте параметр `max_features` леса с помощью `GridSearhCV`, посмотрите на лучшую комбинацию параметров и среднее качество на 5-кратной кросс-валидации. Используйте параметр random_state=17 (для воспроизводимости результатов), не забывайте про распараллеливание (n_jobs=-1).**

# In[28]:


get_ipython().run_cell_magic('time', '', "forest_params = {'max_features': np.linspace(.3, 1, 7)}\n\nlocally_best_forest = GridSearchCV(first_forest, forest_params, cv=5, n_jobs=-1)\nlocally_best_forest.fit(train_df, y)")


# In[29]:


locally_best_forest.best_params_, round(locally_best_forest.best_score_, 3)


# In[30]:


tuned_forest_pred = locally_best_forest.predict_proba(test_df)[:, 1]


# In[31]:


write_to_submission_file(tuned_forest_pred, 'credit_scoring_first_tree_v5.csv')


# **Посмотрите, как настроенный случайный лес оценивает важность признаков по их влиянию на целевой. Представьте результаты в наглядном виде с помощью `DataFrame`.**

# In[32]:


pd.DataFrame(locally_best_forest.best_estimator_.feature_importances_)


# **Обычно увеличение количества деревьев только улучшает результат. Так что напоследок обучите случайный лес из 300 деревьев с найденными лучшими параметрами. Это может занять несколько минут.**

# In[35]:


get_ipython().run_cell_magic('time', '', "final_forest = RandomForestClassifier(n_estimators=300, random_state=17, max_features=0.3, n_jobs=-1)\nfinal_forest.fit(train_df, y)\nfinal_forest_pred = final_forest.predict_proba(test_df)[:, 1]\nwrite_to_submission_file(final_forest_pred, 'credit_scoring_final_forest.csv')")


# **Сделайте посылку на Kaggle.**
