# -*- coding: utf-8 -*-
"""first_lesson_1_home_work_subplots.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m4bRpPNv6BE11Xak5RQVyj_cIQLPHD0S
"""

import pandas as pd
import matplotlib.pyplot as plt
pokemon = pd.read_csv('Pokemon.csv')
pokemon.head(3)

plt.subplots(2, 1, figsize=(8, 8))

fig, axarr = plt.subplots(2, 1, figsize=(8, 8))
pokemon['Attack'].plot.hist(ax=axarr[0], title='Pokemon Attack Ratings')
pokemon['Defense'].plot.hist(ax=axarr[1], title='Pokemon Defense Ratings')