import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


data = pd.read_csv('train.csv')

data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

# select XGBRegressor
my_model = XGBRegressor(n_estimators=750, learning_rate=0.02)
# make pipeline
my_pipeline = make_pipeline(SimpleImputer(), my_model)

scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=2)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))