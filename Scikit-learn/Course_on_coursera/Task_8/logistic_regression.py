# -*- coding: utf-8 -*-
"""logistic_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AHtdMNEwNyloXWsYdjxxtLYFPvUSygSW
"""

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data-logistic.csv', header=None)
df.head()

y = df[0]
X = df.loc[:, 1:]

def fw1(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1

def fw2(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2

def grad(y, X, C=0.0, w1=0.0, w2=0.0, k=0.1, err=1e-5):
    i = 0
    i_max = 10000
    w1_new, w2_new = w1, w2

    while True:
        i += 1
        w1_new, w2_new = fw1(w1, w2, y, X, k, C), fw2(w1, w2, y, X, k, C)
        e = math.sqrt((w1_new - w1) ** 2 + (w2_new - w2) ** 2)

        if i >= i_max or e <= err:
            break
        else:
            w1, w2 = w1_new, w2_new

    return [w1_new, w2_new]

w1, w2 = grad(y, X)
rw1, rw2 = grad(y, X, 10.0)

def a(X, w1, w2):
    return 1.0 / (1.0 + math.exp(-w1 * X[1] - w2 * X[2]))

y_score = X.apply(lambda x: a(x, w1, w2), axis=1)
y_rscore = X.apply(lambda x: a(x, rw1, rw2), axis=1)

auc = roc_auc_score(y, y_score)
rauc = roc_auc_score(y, y_rscore)

ans = "{:0.3f} {:0.3f}".format(auc, rauc)
ans

clf_auc = LogisticRegression(penalty='l1', C=1.0, tol=1e-5, random_state=0, intercept_scaling=0.1)
clf_auc.fit(X, y)
y_score = clf_auc.predict_proba(X)[:, 1:2]
c_auc = roc_auc_score(y, y_score)
c_auc

clf_rauc = LogisticRegression(penalty='l2', C=0.02, tol=1e-5, random_state=0, solver='lbfgs', intercept_scaling=0.1, multi_class='ovr')
clf_rauc.fit(X, y)
y_rscore = clf_rauc.predict_proba(X)[:, 1:2]
c_rauc = roc_auc_score(y, y_rscore)
c_rauc

answer = "{:0.3f} {:0.3f}".format(c_auc, c_rauc)
answer