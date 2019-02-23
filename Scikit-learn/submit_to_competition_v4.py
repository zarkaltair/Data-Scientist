import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
pd.options.display.expand_frame_repr = False

# Path of files
train_data_path = 'train.csv'
test_data_path = 'test.csv'

# read files
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# specifications for model
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

t_X = test_data.select_dtypes(exclude=['object'])

# train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=1)

# use Imputere for missing values
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(X)
test_X = my_imputer.transform(t_X)

# select XGBRegressor
my_model = XGBRegressor(n_estimators=750, learning_rate=0.02)

# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y)

# make predictions which we will submit
test_preds = my_model.predict(t_X)

# The lines below shows how to save predictions in format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)