import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# define target
target = train_data.SalePrice

# Drop houses where the target is missing
iowa_predictors = train_data.dropna(['SalePrice'], axis=1)
# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

# Get Model Score from Imputation
my_imputer = SimpleImputer()
# candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_train_predictors = my_imputer.fit_transform(iowa_numeric_predictors)
# candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)
candidate_test_predictors = my_imputer.transform(test_data)



# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(n_estimators=100, random_state=1), 
                                X, y, 
                                cv=5, 
                                scoring='neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))


# one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
# one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
# final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
#                                                                     join='left', 
#                                                                     axis=1)