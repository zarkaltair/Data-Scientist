# -*- coding: utf-8 -*-
"""supporting_objects.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NBjODAdvdjUXPB26dt7jNHDS8AX8wuzs
"""

import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv('svm-data.csv', header=None)

y = df[0]
x = df.loc[:, 1:]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(x, y)
supports = clf.support_
supports