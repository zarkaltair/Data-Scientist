# -*- coding: utf-8 -*-
"""Titanic_task_for_kaggle_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KWWx_TGC7nf4VHEY41dcKNCDZqAUw1e6
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

train_data = 'train.csv'
test_data = 'test.csv'
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

y = train_df['Survived']
x = train_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

num_feat = x.select_dtypes('number').columns.values
cat_feat = x.select_dtypes('object').columns.values
x_num = x[num_feat]
x_cat = x[cat_feat]

x_num = (x_num - x_num.mean()) / x_num.std()
x_num = x_num.fillna(x_num.mean())

x_cat = pd.get_dummies(x_cat)

x = pd.concat([x_num, x_cat], axis=1)

num_feat = test_x.select_dtypes('number').columns.values
cat_feat = test_x.select_dtypes('object').columns.values
test_x_num = test_x[num_feat]
test_x_cat = test_x[cat_feat]

test_x_num = (test_x_num - test_x_num.mean()) / test_x_num.std()
test_x_num = test_x_num.fillna(test_x_num.mean())

test_x_cat = pd.get_dummies(test_x_cat)

test_x = pd.concat([test_x_num, test_x_cat], axis=1)

dt = DecisionTreeClassifier(random_state=1)
dt.fit(x, y)
predictions = dt.predict(test_x)

output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                       'Survived': predictions})
output.to_csv('submission_v2.csv', index=False)