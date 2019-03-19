#!/usr/bin/env python
# coding: utf-8

# # <center> Домашнее задание № 4 (демо).
# ## <center>  Прогнозирование популярности статей на TechMedia (Хабр) с помощью линейных моделей
#     
# **В задании Вам предлагается разобраться с тем, как работает TfidfVectorizer и DictVectorizer, затем обучить и настроить модель линейной регрессии Ridge на данных о публикациях на Хабрахабре. Пройдя все шаги, вы сможете получить бейзлайн для [соревнования](https://www.kaggle.com/c/howpop-habrahabr-favs-lognorm) (несмотря на old в названии, для этого задания соревнование актуально). 
# Ответьте на все вопросы в этой тетрадке и заполните ответы в [гугл-форме](https://docs.google.com/forms/d/1gPt401drm84N2kdezwGWtPJN_JpaFqXoh6IwlWOslb4).**

# **Описание соревнования**
# 
# Предскажите, как много звездочек наберет статья, зная только ее текст и время публикации
# 
# Необходимо предсказать популярность поста на Хабре по содержанию и времени публикации. Как известно, пользователи Хабра могут добавлять статьи к себе в избранное. Общее количество пользователей, которое это сделали отображается у статьи количеством звездочек. Будем считать, что число звездочек, поставленных статье, наиболее хорошо отражает ее популярность.
# 
# Более формально, в качестве метрики популярности статьи будем использовать долю статей за последний месяц, у которых количество звездочек меньше чем у текущей статьи. А точнее, доле числа звездочек можно поставить в соответствие квантили стандартного распределения, таким образом получаем числовую характеристику популярности статьи. Популярность статьи 0 означает, что статья получила ровно столько звездочек, сколько в среднем получают статьи. И соответственно чем больше звездочек получила статья по сравнению со средним, тем выше это число.

# **Приступим:** импортируем необходимые библиотеки и скачаем данные

# In[14]:


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import scipy

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)


# Скачайте [данные](https://www.kaggle.com/c/howpop-habrahabr-favs-lognorm/data) соревнования.

# In[15]:


train_df = pd.read_csv('howpop_train.csv')
test_df  = pd.read_csv('howpop_test.csv')


# In[3]:


train_df.head(1).T


# In[4]:


train_df.shape, test_df.shape


# Убедимся, что данные отсортированы по признаку `published`

# In[5]:


train_df['published'].apply(lambda ts: pd.to_datetime(ts).value).plot();


# **Чтобы ответить на вопросы 1 и 2, можно использовать [pandas.DataFrame.corr()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html), [pandas.to_datetime()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) и [pandas.Series.value_counts()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html)**

# <font color='red'>Вопрос 1.</font> Есть ли в train_df признаки, корреляция между которыми больше 0.9? Обратите внимание, именно различные признаки - корреляция признака с самим собой естественно больше 0.9 :)
# - да
# - нет
# - не знаю

# In[19]:


train_df.corr(method='pearson')


# In[18]:


sns.heatmap(train_df.corr(method='pearson'), annot=True, fmt='.1f', linewidths='.3');


# <font color='red'>Вопрос 2.</font> В каком году было больше всего публикаций? (Рассматриваем train_df)
# - 2014
# - 2015
# - 2016
# - 2017

# In[16]:


train_df['published'].apply(lambda ts: ts[: 4]).value_counts()


# ## Разбиение на train/valid
# Используем только признаки 'author', 'flow', 'domain' и 'title'

# In[17]:


features = ['author', 'flow', 'domain','title']
train_size = int(0.7 * train_df.shape[0])


# In[18]:


len(train_df), train_size


# In[19]:


X, y = train_df.loc[:, features],  train_df['favs_lognorm'] #отделяем признаки от целевой переменной

X_test = test_df.loc[:, features]


# In[20]:


X_train, X_valid = X.iloc[:train_size, :], X.iloc[train_size:,:]

y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]


# ## Применение TfidfVectorizer
# 
# **TF-IDF** (от англ. TF — term frequency, IDF — inverse document frequency) — статистическая мера, используемая для оценки важности слова в контексте документа, являющегося частью коллекции документов или корпуса. Вес некоторого слова пропорционален количеству употребления этого слова в документе, и обратно пропорционален частоте употребления слова в других документах коллекции. [Подробнее в источнике](https://ru.wikipedia.org/wiki/TF-IDF)
# 
# TfidfVectorizer преобразует тексты в матрицу TF-IDF признаков.
# 
# **Основные параметры TfidfVectorizer в sklearn:**
# - **min_df** - при построении словаря слова, которые встречаются *реже*, чем указанное значение, игнорируются
# - **max_df** - при построении словаря слова, которые встречаются *чаще*, чем указанное значение, игнорируются
# - **analyzer** - определяет, строятся ли признаки по словам или по символам (буквам)
# - **ngram_range** - определяет, формируются ли признаки только из отдельных слов или из нескольких слов (в случае с analyzer='char' задает количество символов). Например, если указать analyzer='word' и ngram_range=(1,3),то признаки будут формироваться из отдельных слов, из пар слов и из троек слов.
# - **stop_words** - слова, которые игнорируются при построении матрицы
# 
# Более подробно с параметрами можно ознакомиться в [документации](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

# **Инициализируйте TfidfVectorizer с параметрами min_df=3, max_df=0.3 и ngram_range=(1, 3).<br />
# Примените метод fit_transform к X_train['title'] и метод transform к X_valid['title'] и X_test['title']**

# <font color='red'>Вопрос 3.</font> Какой размер у полученного словаря?
# - 43789
# - 50624
# - 93895
# - 74378

# In[25]:


vectorizer_title = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 3))

X_train_title = vectorizer_title.fit_transform(X_train['title'])
X_valid_title = vectorizer_title.transform(X_valid['title'])
X_test_title = vectorizer_title.transform(X_test['title'])


# In[27]:


# Можно посмотреть словарь в виде {'термин': индекс признака,...}
dic = vectorizer_title.vocabulary_


# In[28]:


len(dic)


# <font color='red'>Вопрос 4.</font> Какой индекс у слова 'python'?
# - 1
# - 10
# - 9065
# - 15679

# In[29]:


dic['python']


# **Инициализируйте TfidfVectorizer, указав analyzer='char'.<br />
# Примените метод fit_transform к X_train['title'] и метод transform к X_valid['title'] и X_test['title']**

# <font color='red'>Вопрос 5.</font> Какой размер у полученного словаря?
# - 218
# - 510
# - 125
# - 981

# In[30]:


vectorizer_title_ch = TfidfVectorizer(analyzer='char')

X_train_title_ch = vectorizer_title_ch.fit_transform(X_train['title'])
X_valid_title_ch = vectorizer_title_ch.transform(X_valid['title'])
X_test_title_ch = vectorizer_title_ch.transform(X_test['title'])


# In[32]:


# Здесь так же можно посмотреть словарь
# Заметьте насколько отличаются словари для TfidfVectorizer с analyzer='word' и analyzer='char'
dict_char = vectorizer_title_ch.vocabulary_


# In[33]:


len(dict_char)


# ## Работа с категориальными признаками
# 
# Для обработки категориальных признаков 'author', 'flow', 'domain' мы будем использовать DictVectorizer из sklearn.

# In[34]:


feats = ['author', 'flow', 'domain']
X_train[feats][:5]


# Рассмотрим как он работает на примере первых пяти строк

# In[35]:


# сначала заполняем пропуски прочерком
X_train[feats][:5].fillna('-')


# In[36]:


# Преобразуем датафрейм в словарь, где ключами являются индексы объектов (именно для этого мы транспонировали датафрейм),
# а значениями являются словари в виде 'название_колонки':'значение'
X_train[feats][:5].fillna('-').T.to_dict()


# In[37]:


# В DictVectorizer нам нужно будет передать список словарей для каждого объекта в виде 'название_колонки':'значение',
# поэтому используем .values()
X_train[feats][:5].fillna('-').T.to_dict().values()


# In[38]:


# В итоге получается разреженная матрица
dict_vect = DictVectorizer()
dict_vect_matrix = dict_vect.fit_transform(X_train[feats][:5].fillna('-').T.to_dict().values())
dict_vect_matrix


# In[39]:


# Но можно преобразовать ее в numpy array с помощью .toarray()
dict_vect_matrix.toarray()


# In[40]:


# В получившейся матрице 5 строк (по числу объектов) и 9 столбцов
# Далее разберемся почему колонок именно 9
dict_vect_matrix.shape


# Посмотрим сколько уникальных значений в каждой колонке.<br />
# Суммарно их 9 - столько же, сколько и колонок. Это объясняется тем, что для категориальных признаков со строковыми значениями DictVectorizer делает кодирование бинарными признаками - каждому уникальному значению признака соответствует один новый бинарный признак, который равен 1 только в том случае, если в исходной матрице этот признак принимает значение, которому соответствует эта колонка новой матрицы.

# In[41]:


for col in feats:
    print(col,len(X_train[col][:5].fillna('-').unique()))


# Также можно посмотреть что означает каждая колонка полученной матрицы

# In[42]:


# например, самая первая колонка называется 'author=@DezmASter' - то есть принимает значение 1 только если автор @DezmASter
dict_vect.feature_names_


# **Инициализируйте DictVectorizer с параметрами по умолчанию.<br />
# Примените метод fit_transform к X_train[feats] и метод transform к X_valid[feats] и X_test[feats]**

# In[49]:


X_train[feats].head(3)


# In[50]:


X_valid[feats].head(3)


# In[51]:


X_test[feats].head(3)


# In[52]:


vectorizer_feats = DictVectorizer()

X_train_feats = vectorizer_feats.fit_transform(X_train[feats].fillna('-').T.to_dict().values())
X_valid_feats = vectorizer_feats.transform(X_valid[feats].fillna('-').T.to_dict().values())
X_test_feats = vectorizer_feats.transform(X_test[feats].fillna('-').T.to_dict().values())


# In[53]:


X_train_feats.shape


# Соединим все полученные матрицы при помощи scipy.sparse.hstack()

# In[54]:


X_train_new = scipy.sparse.hstack([X_train_title, X_train_feats, X_train_title_ch])
X_valid_new = scipy.sparse.hstack([X_valid_title, X_valid_feats, X_valid_title_ch])
X_test_new =  scipy.sparse.hstack([X_test_title, X_test_feats, X_test_title_ch])


# ## Обучение модели
# 
# Далее будем использовать Ridge, линейную модель с l2-регуляризацией.
# [Документация](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
# 
# Основной параметр Ridge - **alpha, коэффициент регуляризации**. Регуляризация используется для улучшения обобщающей способности модели - прибавляя к функционалу потерь сумму квадратов весов, умноженную на коэффициент регуляризации (та самая alpha), мы штрафуем модель за слишком большие значения весов и не позволяем ей переобучаться. Чем больше этот коээфициент, тем сильнее эффект.

# **Обучите две модели на X_train_new, y_train, задав в первой alpha=0.1 и random_state = 1, а во второй alpha=1.0 и random_state = 1**
# 
# **Рассчитайте среднеквадратичную ошибку каждой модели (mean_squared_error). Сравните значения ошибки на обучающей и тестовой выборках и ответьте на вопросы.**

# <font color='red'>Вопрос 6.</font> Выберите верные утверждения:
# - обе модели показывают одинаковый результат (среднеквадратичная ошибка отличается не больше чем на тысячные), регуляризация ничего не меняет
# - при alpha=0.1 модель переобучается
# - среднеквадратичная ошибка первой модели на тесте меньше
# - при alpha=1.0 у модели обощающая способность лучше, чем у при alpha=0.1

# In[55]:


get_ipython().run_cell_magic('time', '', 'model1 = Ridge(alpha=0.1, random_state=1)\nmodel1.fit(X_train_new, y_train)')


# In[59]:


train_preds1 = model1.predict(X_train_new)
valid_preds1 = model1.predict(X_valid_new)

print('Ошибка на трейне',mean_squared_error(y_train, train_preds1))
print('Ошибка на тесте',mean_squared_error(y_valid, valid_preds1))


# In[57]:


get_ipython().run_cell_magic('time', '', 'model2 = Ridge(alpha=1.0, random_state=1)\nmodel2.fit(X_train_new, y_train)')


# In[58]:


train_preds2 = model2.predict(X_train_new)
valid_preds2 = model2.predict(X_valid_new)

print('Ошибка на трейне',mean_squared_error(y_train, train_preds2))
print('Ошибка на тесте',mean_squared_error(y_valid, valid_preds2))


# ## Baseline
# 
# **Теперь попытаемся получить бейзлайн для соревования - используйте Ridge с параметрами по умолчанию и обучите модель на всех данных - соедините X_train_new X_valid_new (используйте scipy.sparse.vstack()), а целевой переменной будет y.**

# In[61]:


X_full = scipy.sparse.vstack([X_train_new, X_valid_new])


# In[62]:


get_ipython().run_cell_magic('time', '', 'model = Ridge()\n\nmodel.fit(X_full, y)\n\ntest_preds = model.predict(X_test_new)')


# In[64]:


sample_submission = pd.read_csv('sample_submission.csv', 
                                index_col='url')


# In[65]:


sample_submission.head()


# In[66]:


ridge_submission = sample_submission.copy()
ridge_submission['favs_lognorm'] = test_preds
# это будет бейзлайн "Простое решение"
ridge_submission.to_csv('ridge_baseline_v1.csv') 

