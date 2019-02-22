import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor


data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=5)
# select XGBRegressor
my_model = XGBRegressor(n_estimators=706, learning_rate=0.02)
# make pipeline
my_pipeline = make_pipeline(SimpleImputer(), my_model)
# fit model
my_pipeline.fit(train_X, train_y) # , early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False
# make predictions
predictions = my_pipeline.predict(test_X)
# print(MAE)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, test_y)))