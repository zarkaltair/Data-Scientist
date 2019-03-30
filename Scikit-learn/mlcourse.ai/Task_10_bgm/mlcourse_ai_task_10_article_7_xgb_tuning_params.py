#!/usr/bin/env python
# coding: utf-8

# [Соревнование](https://www.kaggle.com/c/forest-cover-type-prediction). 
# Задача учебная. Предлагается предсказывать тип лесного покрытия на участках 30х30 метров Национального заповедника Рузвельта в Колорадо.

# - Elevation (высота) - Elevation in meters
# - Aspect - Aspect in degrees azimuth
# - Slope (наклон) - Slope in degrees
# - Horizontal_Distance_To_Hydrology (горизонтальное расстояние до воды) - Horz Dist to nearest surface water features
# - Vertical_Distance_To_Hydrology (вертикальное расстояние до воды) - Vert Dist to nearest surface water features
# - Horizontal_Distance_To_Roadways (горизонтальное расстояние до дорог) - Horz Dist to nearest roadway
# - Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# - Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# - Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice 
# - Horizontal_Distance_To_Fire_Points (горизонтальное расстояние до центров воспламенения) - Horz Dist to nearest wildfire ignition points 
# - Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# - Soil_Type (тип почвы) - (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

# **Подключаем библиотеки и загружаем данные. Используем [log_progress](https://github.com/alexanderkuk/log-progress) для отслеживания итераций в циклах.**

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# **Файл forest_test.csv можно скачать [отсюда](https://yadi.sk/d/gqsbD1BLzSYvC)**

# In[2]:


train_df = pd.read_csv('../../data/forest_train.csv')
test_df = pd.read_csv('../../data/forest_test.csv')


# In[3]:


def write_to_submission_file(predicted_labels, out_file,
                             target='Cover_Type', index_label="Id", init_index=15121):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(init_index, 
                                                  predicted_labels.shape[0] + init_index),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# **Создаем признаки.**

# In[4]:


train_df['Under_water'] = train_df.Vertical_Distance_To_Hydrology < 0
test_df['Under_water'] = test_df.Vertical_Distance_To_Hydrology < 0


# In[5]:


train_df['EVDtH'] = train_df.Elevation - train_df.Vertical_Distance_To_Hydrology
test_df['EVDtH'] = test_df.Elevation - test_df.Vertical_Distance_To_Hydrology

train_df['EHDtH'] = train_df.Elevation - train_df.Horizontal_Distance_To_Hydrology*0.2
test_df['EHDtH'] = test_df.Elevation - test_df.Horizontal_Distance_To_Hydrology*0.2


# In[6]:


train_df['Distanse_to_Hydrolody'] = (train_df['Horizontal_Distance_To_Hydrology'] ** 2 + 
                                  train_df['Vertical_Distance_To_Hydrology']**2) ** 0.5
test_df['Distanse_to_Hydrolody'] = (test_df['Horizontal_Distance_To_Hydrology']**2 + 
                                 test_df['Vertical_Distance_To_Hydrology']**2) ** 0.5

train_df['Hydro_Fire_1'] = train_df['Horizontal_Distance_To_Hydrology'] +     train_df['Horizontal_Distance_To_Fire_Points']
test_df['Hydro_Fire_1'] = test_df['Horizontal_Distance_To_Hydrology'] +     test_df['Horizontal_Distance_To_Fire_Points']

train_df['Hydro_Fire_2'] = abs(train_df['Horizontal_Distance_To_Hydrology'] -                                 train_df['Horizontal_Distance_To_Fire_Points'])
test_df['Hydro_Fire_2'] = abs(test_df['Horizontal_Distance_To_Hydrology'] -                            test_df['Horizontal_Distance_To_Fire_Points'])

train_df['Hydro_Road_1'] = abs(train_df['Horizontal_Distance_To_Hydrology'] +                             train_df['Horizontal_Distance_To_Roadways'])
test_df['Hydro_Road_1'] = abs(test_df['Horizontal_Distance_To_Hydrology'] +                            test_df['Horizontal_Distance_To_Roadways'])

train_df['Hydro_Road_2'] = abs(train_df['Horizontal_Distance_To_Hydrology'] -                             train_df['Horizontal_Distance_To_Roadways'])
test_df['Hydro_Road_2'] = abs(test_df['Horizontal_Distance_To_Hydrology'] -                            test_df['Horizontal_Distance_To_Roadways'])

train_df['Fire_Road_1'] = abs(train_df['Horizontal_Distance_To_Fire_Points'] +                            train_df['Horizontal_Distance_To_Roadways'])
test_df['Fire_Road_1'] = abs(test_df['Horizontal_Distance_To_Fire_Points'] +                           test_df['Horizontal_Distance_To_Roadways'])

train_df['Fire_Road_2'] = abs(train_df['Horizontal_Distance_To_Fire_Points'] -                            train_df['Horizontal_Distance_To_Roadways'])
test_df['Fire_Road_2'] = abs(test_df['Horizontal_Distance_To_Fire_Points'] -                           test_df['Horizontal_Distance_To_Roadways'])


# In[7]:


y = train_df['Cover_Type']
train_df = train_df.drop(['Cover_Type', 'Id'], axis=1)
test_df = test_df.drop(['Id'], axis=1)


# In[8]:


y = y - 1 # Чтоб классы нумеровались от о до 6


# In[9]:


from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[10]:


def score(params):
    from sklearn.metrics import log_loss
    print("Training with params:")
    print(params)
    params['max_depth'] = int(params['max_depth'])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, params['num_round'])
    predictions = model.predict(dvalid).reshape((X_test.shape[0], 7))
    score = log_loss(y_test, predictions)
    print("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}


# In[11]:


def optimize(trials):
    space = {
             'num_round': 100,
             'learning_rate': hp.quniform('eta', 0.005, 0.05, 0.005),
             'max_depth': hp.quniform('max_depth', 3, 14, 1),
             'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
             'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma': hp.quniform('gamma', 0.5, 1, 0.01),
             'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.05),
             'num_class' : 7,
             'eval_metric': 'merror',
             'objective': 'multi:softprob',
             'nthread' : 4,
             'silent' : 1
             }
    
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=10)
    return best


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    train_df, y, test_size=0.3, random_state=17)


# In[13]:


trials = Trials()
best_params = optimize(trials)
best_params


# In[17]:


best_params['max_depth'] = int(best_params['max_depth'])
best_params['num_class'] = 7
best_params['eval_metric'] = 'merror'
best_params['objective'] = 'multi:softprob'
best_params['nthread'] = 4
best_params['silent'] = 1


# In[18]:


dtrain = xgb.DMatrix(train_df, y)


# In[19]:


get_ipython().run_cell_magic('time', '', 'xgbCvResult = xgb.cv(best_params, dtrain, \n                      num_boost_round=500,  \n                      nfold=3, early_stopping_rounds=50)')


# In[21]:


plt.plot(range(xgbCvResult.shape[0]), xgbCvResult['test-merror-mean'])
plt.plot(range(xgbCvResult.shape[0]), xgbCvResult['train-merror-mean']);


# In[23]:


best_num_round = np.argmin(xgbCvResult['test-merror-mean'])
best_num_round


# In[31]:


get_ipython().run_line_magic('pinfo', 'xgb.train')


# **Сделаем прогноз для всей тестовой выборки.**

# In[32]:


bestXgb = xgb.train(best_params, dtrain, num_boost_round=best_num_round)


# In[33]:


dtest = xgb.DMatrix(test_df)


# In[34]:


xgboost_predict_proba = bestXgb.predict(dtest)


# In[35]:


xgboost_prediction = np.argmax(xgboost_predict_proba,
                                         axis=1)


# **Мы вычитали из целевых меток 1, теперь добавляем.**

# In[36]:


xgboost_prediction += 1


# In[37]:


write_to_submission_file(xgboost_prediction, 
                         'forest_cover_type_xgboost.csv')


# **У такой посылки на Kaggle результат - 0.771.**
