import pandas

from collections import Counter


data = pandas.read_csv('train.csv', index_col='PassengerId')

index_col='PassengerId'

# task 1
print(data['Sex'].value_counts())

# task 2
Survived = data['Survived'].value_counts()
print(round(Survived[1] / Survived.sum(), 4))

# task 3
Pclass_1 = round(data['Pclass'].value_counts()[1] / data['Pclass'].value_counts().sum(), 4)
print(Pclass_1)
print(data['Pclass'].value_counts())
print(data['Pclass'].value_counts()[1])
print(data['Pclass'].value_counts().sum())


# task 4
Age_mean = data['Age'].mean()
print(round(Age_mean, 2))
Age_median = data['Age'].median()
print(round(Age_median, 2))

# task 5


# task 6
most_popular_name = Counter([name.split(' ')[2] for name in data['Name']]).most_common(15)
print(most_popular_name)