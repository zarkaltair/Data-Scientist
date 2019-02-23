import pandas as pd

from xgboost.sklearn import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# Path to files
train_data_path = 'train.csv'
test_data_path = 'test.csv'

# read files using pandas
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Create target object and call it y
y = train_data.SalePrice

# Create X
X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data.select_dtypes(exclude=['object'])

# select XGBRegressor
my_model = XGBRegressor(n_estimators=750, learning_rate=0.02)
# make pipeline
my_pipeline = make_pipeline(SimpleImputer(), my_model)

# make predictions which we will submit.
test_preds = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)

print('Mean Absolute Error %2f' %(-1 * test_preds.mean()))