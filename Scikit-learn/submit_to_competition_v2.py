import pandas as pd

from xgboost.sklearn import XGBRegressor


# Path to files
train_data_path = 'train.csv'
test_data_path = 'test.csv'

# read files using pandas
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Create target object and call it y
y = train_data.SalePrice

# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data[features]

# To improve accuracy, create a new XGBoost model which you will train on all training data
xgb_model_on_full_data = XGBRegressor(n_estimators=100, random_state=1)

# fit rf_model_on_full_data on all data from the training data
xgb_model_on_full_data.fit(X, y)

# make predictions which we will submit.
test_preds = xgb_model_on_full_data.predict(test_X)