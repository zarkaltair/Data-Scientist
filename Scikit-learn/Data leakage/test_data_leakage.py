import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv('AER_credit_card_data.csv', 
                   true_values = ['yes'],
                   false_values = ['no'])

y = data.card
X = data.drop(['card'], axis=1)

# Since there was no preprocessing, we didn't need a pipeline here. Used anyway as best practice
modeling_pipeline = make_pipeline(RandomForestClassifier(n_estimators=1000))
cv_scores = cross_val_score(modeling_pipeline, X, y, scoring='accuracy', cv=5)
print("Cross-val accuracy: %f" %cv_scores.mean())

# Leaky Predictors
expenditures_cardholders = data.expenditure[data.card]
expenditures_noncardholders = data.expenditure[~data.card]

print('Fraction of those who received a card with no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who received a card with no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))

# Leaky Validation Strategies
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
cv_scores = cross_val_score(modeling_pipeline, X2, y, scoring='accuracy', cv=5)
print("Cross-val accuracy: %f" %cv_scores.mean())