# -*- coding: utf-8 -*-
"""Titanic_task.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LRDmp6mCB3CNex2DAo1bTmicBs6j5hjL
"""

import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv('titanic.csv')

df['Sex'] = df['Sex'] == 'male'

df.dropna(axis=0, inplace=True)

y = df.Survived
features = ['Pclass', 'Fare', 'Age', 'Sex']
x = df[features]
x.head()

clf = tree.DecisionTreeClassifier(random_state=241)
clf = clf.fit(x, y)

importances = clf.feature_importances_

importances