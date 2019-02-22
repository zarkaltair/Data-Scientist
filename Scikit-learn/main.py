import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pd.options.display.max_rows = 10
pd.options.display.max_columns = 100
pd.options.display.expand_frame_repr = False

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)
# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))


# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# define target
y = home_data.SalePrice
iowa_predictors = home_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
X = iowa_predictors.select_dtypes(exclude=['object'])

# Split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=1)

 
# Create function to compare all solutions
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    predictors = model.predict(X_test)
    return mean_absolute_error(predictors, y_test)


# Get Model Score from Dropping Columns with Missing Values
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
drop_column_mae = score_dataset(reduced_X_train, reduced_X_test, y_train, y_test)
print("Validation MAE for Random Forest Model from dropping columns with Missing Values: {:,.0f}".format(drop_column_mae))


# Get Model Score from Imputation
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
imputation_mae = score_dataset(imputed_X_train, imputed_X_test, y_train, y_test)
print("Validation MAE for Random Forest Model from Imputation: {:,.0f}".format(imputation_mae))


# Get Score from Imputation with Extra Columns Showing What Was Imputed
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

extra_imputation_mae = score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test)
print("Validation MAE for Random Forest Model from Imputation while Track What Was Imputed: {:,.0f}".format(extra_imputation_mae))