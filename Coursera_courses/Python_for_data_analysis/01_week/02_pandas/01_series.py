import pandas as pd


s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)

d = {'Moscow': 1000, 'London': 300, 'New York': 150, 'Barcelona': None}
cities = pd.Series(d)
print(cities)
print(cities['Moscow'])
print(cities[['Moscow', 'London']])
print(cities < 1000)
print(cities[cities < 1000])
cities['Moscow'] = 100
print(cities)
cities[cities < 1000] = 3
print(cities)
print(cities * 3)
print(cities[cities.isnull()])
print(cities[cities.notnull()])
