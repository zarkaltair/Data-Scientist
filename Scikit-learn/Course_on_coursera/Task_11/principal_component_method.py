# -*- coding: utf-8 -*-
"""principal_component_method.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z_pyqI6hLnR2kvJs7bUl5KC24url4Ye7
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

close_prices = pd.read_csv('close_prices.csv')

x = close_prices.drop(['date'], axis=1)

pca = PCA(n_components=10)
pca.fit(x)
pca.explained_variance_ratio_

df = pd.DataFrame(pca.transform(x))
y = df[0]

dj = pd.read_csv('djia_index.csv')
y_test = dj['^DJI']

corr = np.corrcoef(y, y_test)
corr

pca.explained_variance_

comp_0 = pd.Series(pca.components_[0])
comp_0_top = comp_0.sort_values(ascending=False).head(1).index[0]
x.columns[comp_0_top]