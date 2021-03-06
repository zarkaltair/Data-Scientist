# -*- coding: utf-8 -*-
"""first_lesson_home_work_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17z2eYAMXU5svbEpFP3vaT2WkXSleRlXl
"""

import pandas as pd
import seaborn as sns
pokemon = pd.read_csv("Pokemon.csv", index_col=0)
pokemon.head()

sns.countplot(pokemon['Generation'])

sns.distplot(pokemon['HP'])

sns.jointplot(x='Attack', y='Defense', data=pokemon)

sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex', gridsize=20)

sns.kdeplot(pokemon['HP'], pokemon['Attack'])

sns.boxplot(x='Legendary', y='Attack', data=pokemon)

sns.violinplot(x='Legendary', y='Attack', data=pokemon)