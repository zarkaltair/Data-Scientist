import pandas as pd
import numpy as np


df = pd.read_csv('citibike.csv')

print(df.groupby(['usertype']).groups)
print(df.groupby(['usertype']).first())
print(df.groupby(['usertype'])[['tripduration']].mean())
print(df.groupby(['usertype', 'start station name'])[['tripduration']].mean())
print(df.groupby(['usertype']).agg({'tripduration': sum, 'starttime': 'first'}))
print(df.groupby(['usertype']).agg({'tripduration': [sum, min], 'starttime': 'first'}))
print(df.groupby(['usertype']).agg({'tripduration': lambda x: max(x) + 1, 'starttime': 'first'}))
