#!/usr/bin/env python
# coding: utf-8

# # <center>Домашнее задание № 7 (демо)
# ## <center> Обучение без учителя: метод главных компонент и кластеризация

# В этом задании мы разберемся с тем, как работают методы снижения размерности и кластеризации данных. Заодно еще раз попрактикуемся в задаче классификации.
# 
# Мы будем работать с набором данных [Samsung Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones). Скачайте данные [отсюда](https://drive.google.com/file/d/14RukQ0ylM2GCdViUHBBjZ2imCaYcjlux/view?usp=sharing). Данные поступают с акселерометров и гироскопов мобильных телефонов Samsung Galaxy S3 (подробнее про признаки – по ссылке на UCI выше), также известен вид активности человека с телефоном в кармане – ходил ли он, стоял, лежал, сидел или шел вверх/вниз по лестнице. 
# 
# Вначале мы представим, что вид активности нам неизвестнен, и попробуем кластеризовать людей чисто на основе имеющихся признаков. Затем решим задачу определения вида физической активности именно как задачу классификации. 
# 
# Заполните код в клетках (где написано "Ваш код здесь") и ответьте на вопросы в [веб-форме](https://docs.google.com/forms/d/1qzcrfsNFy-e4TW59v2fqMj_OTom2SIOxtq4MWlI92p0).

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use(['seaborn-darkgrid'])
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 17


# In[3]:


X_train = np.loadtxt("samsung_train.txt")
y_train = np.loadtxt("samsung_train_labels.txt").astype(int)

X_test = np.loadtxt("samsung_test.txt")
y_test = np.loadtxt("samsung_test_labels.txt").astype(int)


# In[4]:


# Проверим размерности
assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))
assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))


# Для кластеризации нам не нужен вектор ответов, поэтому будем работать с объединением обучающей и тестовой выборок. Объедините *X_train* с *X_test*, а *y_train* – с *y_test*. 

# In[10]:


X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
X.shape, y.shape


# Определим число уникальных значений меток целевого класса.

# In[11]:


np.unique(y)


# In[12]:


n_classes = np.unique(y).size


# [Эти метки соответствуют:](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names)
# - 1 - ходьбе
# - 2 - подъему вверх по лестнице
# - 3 - спуску по лестнице
# - 4 - сидению
# - 5 - стоянию
# - 6 - лежанию
# 
# *уж простите, если звучание этих существительных кажется корявым :)*

# Отмасштабируйте выборку с помощью `StandardScaler` с параметрами по умолчанию.

# In[13]:


# Ваш код здесь
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Понижаем размерность с помощью PCA, оставляя столько компонент, сколько нужно для того, чтобы объяснить как минимум 90% дисперсии исходных (отмасштабированных) данных. Используйте отмасштабированную выборку и зафиксируйте random_state (константа RANDOM_STATE).

# In[14]:


# Ваш код здесь
pca = PCA(random_state=RANDOM_STATE)
X_pca = pca.fit(X_scaled)


# **Вопрос 1:**<br>
# Какое минимальное число главных компонент нужно выделить, чтобы объяснить 90% дисперсии исходных (отмасштабированных) данных?

# In[19]:


plt.figure(figsize=(12, 8))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 200)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(65, c='b')
plt.axhline(0.9, c='r')
plt.show();


# **Варианты:**
# - 56 
# - 65
# - 66
# - 193

# **Вопрос 2:**<br>
# Сколько процентов дисперсии приходится на первую главную компоненту? Округлите до целых процентов. 
# 
# **Варианты:**
# - 45
# - 51
# - 56
# - 61

# In[22]:


for i, component in enumerate(pca.components_):
    print("{} component: {}% of initial variance".format(i + 1, 
          round(100 * pca.explained_variance_ratio_[i], 2)))


# Визуализируйте данные в проекции на первые две главные компоненты.

# In[25]:


X_reduced = pca.fit_transform(X)

plt.figure(figsize=(16,12))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=20, cmap='viridis');


# **Вопрос 3:**<br>
# Если все получилось правильно, Вы увидите сколько-то кластеров, почти идеально отделенных друг от друга. Какие виды активности входят в эти кластеры?<br>
# 
# **Ответ:**
# - 1 кластер: все 6 активностей
# - 2 кластера: (ходьба, подъем вверх по лестнице, спуск по лестнице) и (сидение, стояние, лежание)
# - 3 кластера: (ходьба), (подъем вверх по лестнице, спуск по лестнице) и (сидение, стояние, лежание)
# - 6 кластеров

# ------------------------------

# Сделайте кластеризацию данных методом `KMeans`, обучив модель на данных со сниженной за счет PCA размерностью. В данном случае мы подскажем, что нужно искать именно 6 кластеров, но в общем случае мы не будем знать, сколько кластеров надо искать.
# 
# Параметры:
# 
# - **n_clusters** = n_classes (число уникальных меток целевого класса)
# - **n_init** = 100
# - **random_state** = RANDOM_STATE (для воспроизводимости результата)
# 
# Остальные параметры со значениями по умолчанию.

# In[52]:


kmeans = KMeans(n_clusters=n_classes, n_init=100, random_state=RANDOM_STATE, n_jobs=-1)
kmeans.fit(X_reduced)


# Визуализируйте данные в проекции на первые две главные компоненты. Раскрасьте точки в соответствии с полученными метками кластеров.

# In[53]:


plt.figure(figsize=(16,12))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, s=20,  cmap='viridis');


# Посмотрите на соответствие между метками кластеров и исходными метками классов и на то, какие виды активностей алгоритм `KMeans` путает.

# In[31]:


tab = pd.crosstab(y, kmeans.labels_, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице', 
             'спуск по лестнице', 'сидение', 'стояние', 'лежание', 'все']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['все']
tab


# Видим, что каждому классу (т.е. каждой активности) соответствуют несколько кластеров. Давайте посмотрим на максимальную долю объектов в классе, отнесенных к какому-то одному кластеру. Это будет простой метрикой, характеризующей, насколько легко класс отделяется от других при кластеризации. 
# 
# Пример: если для класса "спуск по лестнице", в котором 1406 объектов,  распределение кластеров такое:
#  - кластер 1 – 900
#  - кластер 3 – 500
#  - кластер 6 – 6,
#  
# то такая доля будет 900 / 1406 $\approx$ 0.64.
#  
# 
# **Вопрос 4:**<br>
# Какой вид активности отделился от остальных лучше всего в терминах простой  метрики, описанной выше?<br>
# 
# **Ответ:**
# - ходьба
# - стояние
# - спуск по лестнице
# - перечисленные варианты не подходят

# Видно, что kMeans не очень хорошо отличает только активности друг от друга. Используйте метод локтя, чтобы выбрать оптимальное количество кластеров. Параметры алгоритма и данные используем те же, что раньше, меняем только `n_clusters`.

# In[32]:


inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))


# In[36]:


plt.plot(range(1, 7), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');


# **Вопрос 5:**<br>
# Какое количество кластеров оптимально выбрать, согласно методу локтя?<br>
# 
# **Ответ:**
# - 1
# - 2
# - 3
# - 4

# ------------------------

# Попробуем еще один метод кластеризации, который описывался в статье – агломеративную кластеризацию.

# In[41]:


ag = AgglomerativeClustering(n_clusters=n_classes, linkage='ward')
ag.fit(X)


# Посчитайте Adjusted Rand Index (`sklearn.metrics`) для получившегося разбиения на кластеры и для `KMeans` с параметрами из задания к 4 вопросу.

# In[42]:


metrics.adjusted_rand_score(y, ag.labels_)


# In[43]:


metrics.adjusted_rand_score(y, kmeans.labels_)


# **Вопрос 6:**<br>
# Отметьте все верные утверждения.<br>
# 
# **Варианты:**
# - Согласно ARI, KMeans справился с кластеризацией хуже, чем Agglomerative Clustering
# - Для ARI не имеет значения какие именно метки присвоены кластерам, имеет значение только разбиение объектов на кластеры
# - В случае случайного разбиения на кластеры ARI будет близок к нулю

# -------------------------------

# Можно заметить, что задача не очень хорошо решается именно как задача кластеризации, если выделять несколько кластеров (> 2). Давайте теперь решим задачу классификации, вспомнив, что данные у нас размечены.  
# 
# Для классификации используйте метод опорных векторов – класс `sklearn.svm.LinearSVC`. Мы в курсе отдельно не рассматривали этот алгоритм, но он очень известен, почитать про него можно, например, в материалах Евгения Соколова –  [тут](https://github.com/esokolov/ml-course-msu/blob/master/ML16/lecture-notes/Sem11_linear.pdf). 
# 
# Настройте для `LinearSVC` гиперпараметр `C` с помощью `GridSearchCV`. 
# 
# - Обучите новый `StandardScaler` на обучающей выборке (со всеми исходными признаками), прмиените масштабирование к тестовой выборке
# - В `GridSearchCV` укажите  cv=3.

# In[44]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[45]:


svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}


# In[47]:


best_svc = GridSearchCV(svc, svc_params, cv=3, n_jobs=-1)
best_svc.fit(X_train_scaled, y_train)


# In[54]:


best_svc.best_params_, best_svc.best_score_


# **Вопрос 7**<br>
# Какое значение гиперпараметра `C` было выбрано лучшим по итогам кросс-валидации?<br>
# 
# **Ответ:**
# - 0.001
# - 0.01
# - 0.1
# - 1
# - 10

# In[49]:


y_predicted = best_svc.predict(X_test_scaled)


# In[51]:


tab = pd.crosstab(y_test, y_predicted, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице', 'спуск по лестнице', 
             'сидение', 'стояние', 'лежание', 'все']
tab.columns = tab.index
tab


# **Вопрос 8:**<br>
# Какой вид активности SVM определяет хуже всего в терминах точности? Полноты? <br>
# 
# **Ответ:**
# - по точности – подъем вверх по лестнице, по полноте – лежание
# - по точности – лежание, по полноте – сидение
# - по точности – ходьба, по полноте – ходьба
# - по точности – стояние, по полноте – сидение 

# In[55]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.9, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[56]:


svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}


# In[57]:


get_ipython().run_cell_magic('time', '', 'best_svc_pca = GridSearchCV(svc, svc_params, n_jobs=1, cv=3, verbose=1)\nbest_svc_pca.fit(X_train_pca, y_train);')


# In[58]:


best_svc_pca.best_params_, best_svc_pca.best_score_


# In[59]:


round(100 * (best_svc_pca.best_score_ - best_svc.best_score_))


# Наконец, проделайте то же самое, что в 7 вопросе, только добавив PCA.
# 
# - Используйте выборки `X_train_scaled` и `X_test_scaled`
# - Обучите тот же PCA, что раньше, на отмасшабированной обучающей выборке, примените преобразование к тестовой
# - Настройте гиперпараметр `C` на кросс-валидации по обучающей выборке с PCA-преобразованием. Вы заметите, насколько это проходит быстрее, чем раньше.
# 
# **Вопрос 9:**<br>
# Какова разность между лучшим качеством (долей верных ответов) на кросс-валидации в случае всех 561 исходных признаков и во втором случае, когда применялся метод главных компонент? Округлите до целых процентов.<br>
# 
# **Варианты:**
# - Качество одинаковое
# - 2%
# - 4% 
# - 10%
# - 20%
# 

# **Вопрос 10:**<br>
# Выберите все верные утверждения:
# 
# **Варианты:**
# - Метод главных компонент в данном случае позволил уменьшить время обучения модели, при этом качество (доля верных ответов на кросс-валидации) очень пострадало, более чем на 10%
# - PCA можно использовать для визуализации данных, однако для этой задачи есть и лучше подходящие методы, например, tSNE. Зато PCA имеет меньшую вычислительную сложность
# - PCA строит линейные комбинации исходных признаков, и в некоторых задачах они могут плохо интерпретироваться человеком
