import pandas as pd
import numpy as np


df = pd.read_csv('citibike.csv')

print(df.head(3))
print(df.tail(2))
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df[['starttime', 'start station name']].head())
print(df.iloc[-1])
print(df.iloc[-1, 4])
print(df.loc[1, ['tripduration']])
print(df.iloc[0: 6, 0: 4])
print(df.loc[0: 6, 'tripduration': 'start station name'])

print(df[df['tripduration'] < 400].shape)
print(df[(df['tripduration'] < 1000) & (df['usertype'] == 'Subscriber')].shape)

print(df.describe())
print(df.describe(include=[np.object]))

print(df['usertype'].value_counts())
print(df['usertype'].value_counts(normalize=True))
print(df['gender'].unique())

print(df.corr())

print(df.sample(frac=0.1))

df.to_csv('path_to_file.csv')
