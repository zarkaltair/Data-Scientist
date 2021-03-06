# -*- coding: utf-8 -*-
"""final_test_logreg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S5e0sF86T2T-C98r5pf9MFCzlrTjnd4T
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# pd.set_option('max_rows', None)
df = pd.read_csv('features.csv', index_col='match_id')

df_test = pd.read_csv('features_test.csv', index_col='match_id')
df_test.shape

# passes have the following signs:
passes = ['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 
         'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time', 'dire_bottle_time', 
         'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']
# 'first_blood_time' - time first kill
# 'first_blood_team' - team which make first kill

df[passes] = df[passes].fillna(0)
df_test[passes] = df_test[passes].fillna(0)

# target variable
y = df['radiant_win']
X = df.drop(['radiant_win', 'duration', 'tower_status_radiant', 'tower_status_dire', 
             'barracks_status_radiant', 'barracks_status_dire'], axis=1)
X.shape

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X)
X_test_scaler = scaler.transform(df_test)

logreg_model = LogisticRegression(penalty='l2', C=0.01, random_state=42, solver='lbfgs')
logreg_model.fit(X_train_scaler, y)
score = cross_val_score(logreg_model, X_train_scaler, y, cv=kfold, scoring='roc_auc')
score.mean()
# y_score = logreg_model.predict_proba(X_train_scaler)
# scores = roc_auc_score(y, y_score[:, 1])
# scores

c_variables = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
               'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
X_train_cc = X.drop(c_variables, axis=1)
X_test_cc = df_test.drop(c_variables, axis=1)

X_train_scaler_cc = scaler.fit_transform(X_train_cc)
X_test_scaler_cc = scaler.transform(X_test_cc)

logreg_model_cc = LogisticRegression(penalty='l2', C=0.01, random_state=42, solver='lbfgs')
logreg_model_cc.fit(X_train_scaler_cc, y)
score_cc = cross_val_score(logreg_model_cc, X_train_scaler_cc, y, cv=kfold, scoring='roc_auc')
score_cc.mean()

heroes = pd.read_csv('heroes.csv')
len(heroes)

X_pick = np.zeros((df.shape[0], len(heroes)))
for i, match_id in enumerate(df.index):
    for p in range(5):
        X_pick[i, df.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, df.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick_df = pd.DataFrame(X_pick, index=X.index)

X_pick_df.shape

X_train_cc_scaler = pd.DataFrame(scaler.fit_transform(X_train_cc), index = X.index)
X_concat = pd.concat([X_train_cc_scaler, X_pick_df], axis=1)

logreg_model_concat = LogisticRegression(penalty='l2', C=1.0, random_state=42, solver='lbfgs')
logreg_model_concat.fit(X_concat, y)
score_concat = cross_val_score(logreg_model_concat, X_concat, y, cv=kfold, scoring='roc_auc')
score_concat.mean()

X_pick_test = np.zeros((df_test.shape[0], len(heroes)))
for i, match_id in enumerate(df_test.index):
    for p in range(5):
        X_pick_test[i, df_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, df_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick_test_df = pd.DataFrame(X_pick_test, index=df_test.index)

X_test_cc_scaler = pd.DataFrame(scaler.fit_transform(X_test_cc), index = df_test.index)
X_test_concat = pd.concat([X_test_cc_scaler, X_pick_test_df], axis=1)

y_test = logreg_model_concat.predict_proba(X_test_concat)[:, 1]
submission = pd.DataFrame({'radiant_win': y_test}, index=X_test_concat.index)
submission.index.name = 'match_id'
submission.to_csv('submission.csv', index=False)