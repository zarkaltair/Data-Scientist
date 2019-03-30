#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.max.columns', 100)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Считаем данные и посмотрим на первые несколько строк. Видим, что у нас тут немало категориальных признаков.

# In[3]:


df = pd.read_csv('bank.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# Всего 9 признаков со строковыми значениями.

# In[6]:


df.columns[df.dtypes == 'object']


# ## Без категориальных признаков
# Попытаемся сначала просто проигнорировать категориальные признаки. Обучим случайный лес и посмотрим на ROC AUC на кросс-валидации и на отоженной выборке. Это будет наш бейзлайн. 

# In[7]:


df_no_cat, y = df.loc[:, df.dtypes != 'object'].drop('y', axis=1), df['y']


# In[8]:


df_no_cat_part, df_no_cat_valid, y_train_part, y_valid = train_test_split(df_no_cat, y,
                                                                          test_size=.3, 
                                                                          stratify=y, 
                                                                          random_state=17)


# In[9]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


# In[12]:


forest = RandomForestClassifier(n_estimators=100, random_state=17)


# In[13]:


np.mean(cross_val_score(forest, df_no_cat_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[14]:


forest.fit(df_no_cat_part, y_train_part)


# In[15]:


roc_auc_score(y_valid, forest.predict_proba(df_no_cat_valid)[:, 1])


# ## LabelEncoder для категориальных признаков
# Сделаем то же самое, но попробуем закодировать категориальные признаки по-простому: с помощью `LabelEncoder`.

# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


label_encoder = LabelEncoder()


# In[18]:


df_cat_label_enc = df.copy().drop('y', axis=1)
for col in df.columns[df.dtypes == 'object']:
    df_cat_label_enc[col] = label_encoder.fit_transform(df_cat_label_enc[col])


# In[19]:


df_cat_label_enc.shape


# In[20]:


df_cat_label_enc_part, df_cat_label_enc_valid = train_test_split(df_cat_label_enc, test_size=.3, 
                                                    stratify=y, random_state=17)


# In[21]:


np.mean(cross_val_score(forest, df_cat_label_enc_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[22]:


forest.fit(df_cat_label_enc_part, y_train_part)


# In[24]:


roc_auc_score(y_valid, forest.predict_proba(df_cat_label_enc_valid)[:, 1])


# ## Бинаризация категориальных признаков (dummies, OHE)
# Теперь сделаем то, что обычно по умолчанию и делают – бинаризацию категориальных признаков. Dummy-признаки, One-Hot Encoding... с небольшими различиями это об одном же - для каждого значения каждого категориального признака завести свой бинарный признак.

# In[25]:


df_cat_dummies = pd.get_dummies(df, columns=df.columns[df.dtypes == 'object']).drop('y', axis=1)


# In[26]:


df_cat_dummies.shape


# In[27]:


df_cat_dummies_part, df_cat_dummies_valid = train_test_split(df_cat_dummies, test_size=.3, 
                                                    stratify=y, random_state=17)


# In[28]:


np.mean(cross_val_score(forest, df_cat_dummies_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[29]:


forest.fit(df_cat_dummies_part, y_train_part)


# In[30]:


roc_auc_score(y_valid, forest.predict_proba(df_cat_dummies_valid)[:, 1])


# ## Попарные взаимодействия признаков
# Пока лес все еще лучше регрессии (хотя мы не тюнили гиперпараметры, но и не будем). Мы хотим идти дальше. Мощной техникой для работы с категориальными признаками будет учет попарных взаимодействий признаков (feature interactions). Построим попарные взаимодействия всех признаков. Вообще тут можно пойти дальше и строить взаимодействия трех и более признаков. Owen Zhang [как-то строил](https://www.youtube.com/watch?v=LgLcfZjNF44) даже 7-way interactions. Чего не сделаешь ради победы на Kaggle! :)

# In[31]:


df_interact = df.copy()


# In[32]:


cat_features = df.columns[df.dtypes == 'object']
for i, col1 in enumerate(cat_features):
    for j, col2 in enumerate(cat_features[i + 1:]):
        df_interact[col1 + '_' + col2] = df_interact[col1] + '_' + df_interact[col2] 


# In[33]:


df_interact.shape


# In[34]:


df_interact.head()


# ## Бинаризация категориальных признаков (dummies, OHE) + попарные взаимодействия
# Получилось аж 824 бинарных признака – многовато для такой задачи, и тут случайный лес начинает не справляться, да и логистическая регрессия сработала хуже, чем в прошлый раз.

# In[35]:


df_interact_cat_dummies = pd.get_dummies(df_interact, columns=df_interact.columns[df_interact.dtypes == 'object']).drop('y', axis=1)


# In[36]:


df_interact_cat_dummies.shape


# In[37]:


df_interact_cat_dummies_part, df_interact_cat_dummies_valid = train_test_split(df_interact_cat_dummies, test_size=.3, 
                                                    stratify=y, random_state=17)


# In[38]:


np.mean(cross_val_score(forest, df_interact_cat_dummies_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[39]:


forest.fit(df_interact_cat_dummies_part, y_train_part)


# In[40]:


roc_auc_score(y_valid, forest.predict_proba(df_interact_cat_dummies_valid)[:, 1])


# Случайному лесу уже тяжеловато, когда признаков так много, а вот логистической регрессии – норм.

# In[73]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(solver='lbfgs', max_iter=500, random_state=17)


# In[74]:


np.mean(cross_val_score(logit, df_interact_cat_dummies_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[75]:


logit.fit(df_interact_cat_dummies_part, y_train_part)


# In[76]:


roc_auc_score(y_valid, logit.predict_proba(df_interact_cat_dummies_valid)[:, 1])


# ## Mean Target
# Теперь будем использовать технику кодирования категориальных признаков средним значением целевого признака. Это очень мощная техника, правда, надо умело ее использовать – легко переобучиться. 
# Основная идея – для каждого значения категориального признака посчитать среднее значение целевого признака и заменить категориальный признак на посчитанные средние. Правда, считать средние надо на кросс-валидации, а то легко переобучиться. 
# Но далее я адресую к видео топ-участников соревнований Kaggle, от них можно узнать про эту технику из первых уст. 
# - [Специализация](https://www.coursera.org/specializations/aml) "Advanced Machine Learning" на Coursera, [курс](https://www.coursera.org/learn/competitive-data-science)", How to Win a Data Science Competition: Learn from Top Kagglers", несколько видео посвящено различным способам построяния признаков с задействованием целевого, и как при этом не переобучиться. Рассказывает Дмитрий Алтухов
# - [Лекция](https://www.youtube.com/watch?v=g335THJxkto) с презентацией решения конкурса Kaggle BNP paribas, Станислав Семенов
# 
# Похожая техника [используется](https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/) и в CatBoost.
# 
# Для начала давайте таким образом закодируем исходные категориальные признаки.

# In[47]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# In[48]:


train_df, y = df.copy(), df['y']
train_df_part, valid_df, y_train_part, y_valid = train_test_split(train_df.drop('y', axis=1), y, 
                                                                  test_size=.3, stratify=y, 
                                                                               random_state=17)


# In[49]:


def mean_target_enc(train_df, y_train, valid_df, skf):
    import warnings
    warnings.filterwarnings('ignore')
    
    glob_mean = y_train.mean()
    train_df = pd.concat([train_df, pd.Series(y_train, name='y')], axis=1)
    new_train_df = train_df.copy()
    
    cat_features = train_df.columns[train_df.dtypes == 'object'].tolist()    

    for col in cat_features:
        new_train_df[col + '_mean_target'] = [glob_mean for _ in range(new_train_df.shape[0])]

    for train_idx, valid_idx in skf.split(train_df, y_train):
        train_df_cv, valid_df_cv = train_df.iloc[train_idx, :], train_df.iloc[valid_idx, :]

        for col in cat_features:
            
            means = valid_df_cv[col].map(train_df_cv.groupby(col)['y'].mean())
            valid_df_cv[col + '_mean_target'] = means.fillna(glob_mean)
            
        new_train_df.iloc[valid_idx] = valid_df_cv
    
    new_train_df.drop(cat_features + ['y'], axis=1, inplace=True)
    
    for col in cat_features:
        means = valid_df[col].map(train_df.groupby(col)['y'].mean())
        valid_df[col + '_mean_target'] = means.fillna(glob_mean)
        
    valid_df.drop(train_df.columns[train_df.dtypes == 'object'], axis=1, inplace=True)
    
    return new_train_df, valid_df


# In[50]:


train_mean_target_part, valid_mean_target = mean_target_enc(train_df_part, y_train_part, valid_df, skf)


# In[51]:


np.mean(cross_val_score(forest, train_mean_target_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[52]:


forest.fit(train_mean_target_part, y_train_part)


# In[53]:


roc_auc_score(y_valid, forest.predict_proba(valid_mean_target)[:, 1])


# ## Mean Target + попарные взаимодействия

# In[54]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# In[55]:


train_df, y = df_interact.drop('y', axis=1).copy(), df_interact['y']
train_df_part, valid_df, y_train_part, y_valid = train_test_split(train_df, y, 
                                                                  test_size=.3, stratify=y, 
                                                                  random_state=17)


# In[56]:


train_mean_target_part, valid_mean_target = mean_target_enc(train_df_part, y_train_part, valid_df, skf)


# In[57]:


np.mean(cross_val_score(forest, train_mean_target_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[58]:


forest.fit(train_mean_target_part, y_train_part)


# In[59]:


roc_auc_score(y_valid, forest.predict_proba(valid_mean_target)[:, 1])


# Опять лучше справляется логистическая регрессия.

# In[77]:


np.mean(cross_val_score(logit, train_mean_target_part, y_train_part, cv=skf, scoring='roc_auc'))


# In[78]:


logit.fit(train_mean_target_part, y_train_part)


# In[79]:


roc_auc_score(y_valid, logit.predict_proba(valid_mean_target)[:, 1])


# ## Catboost
# В библиотеке [Catboost](https://catboost.yandex), помимо всего прочего, реализована как раз техника кодирования категориальных значений средним значением целевого признака. Результаты получаются хорошими именно когда в данных много важных категориальных признаков. Из минусов можно отметить меньшую (пока что) производительность в сравнении с Xgboost и LightGBM.

# In[64]:


from catboost import CatBoostClassifier


# In[65]:


ctb = CatBoostClassifier(random_seed=17)


# In[66]:


train_df, y = df.drop('y', axis=1), df['y']
train_df_part, valid_df, y_train_part, y_valid = train_test_split(train_df, y, 
                                                                  test_size=.3, stratify=y, 
                                                                  random_state=17)


# In[67]:


cat_features_idx = np.where(train_df_part.dtypes == 'object')[0].tolist()


# In[68]:


get_ipython().run_cell_magic('time', '', 'cv_scores = []\nfor train_idx, test_idx in skf.split(train_df_part, y_train_part):\n    cv_train_df, cv_valid_df = train_df_part.iloc[train_idx, :], train_df_part.iloc[test_idx, :]\n    y_cv_train, y_cv_valid = y_train_part.iloc[train_idx], y_train_part.iloc[test_idx]\n    \n    ctb.fit(cv_train_df, y_cv_train,\n        cat_features=cat_features_idx);\n    \n    cv_scores.append(roc_auc_score(y_cv_valid, ctb.predict_proba(cv_valid_df)[:, 1]))')


# In[69]:


np.mean(cv_scores)


# In[70]:


get_ipython().run_cell_magic('time', '', 'ctb.fit(train_df_part, y_train_part,\n        cat_features=cat_features_idx);')


# In[71]:


roc_auc_score(y_valid, ctb.predict_proba(valid_df)[:, 1])

