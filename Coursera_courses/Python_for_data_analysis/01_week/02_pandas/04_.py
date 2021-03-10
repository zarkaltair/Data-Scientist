import pandas as pd
import numpy as np


df = pd.read_csv('citibike.csv')
df.head()

usertype = {'Customer': 1, 'Subscriber': 2}
print(df['usertype'].map(usertype).head())

print(df.apply(min))
print(df['tripduration'].apply(lambda x: x / 60).head())
print(df.apply(lambda x: x['tripduration'] / 60, axis=1).head())
