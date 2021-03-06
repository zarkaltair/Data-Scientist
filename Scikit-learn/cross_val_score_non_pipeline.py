import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score


data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)

my_model = XGBRegressor(n_estimators=750, learning_rate=0.02)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(my_model, X, y, scoring='neg_mean_absolute_error', cv=kfold, n_jobs=2)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))