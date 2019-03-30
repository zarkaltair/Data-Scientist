#!/usr/bin/env python
# coding: utf-8

# # Библиотека Vowpal Wabbit

# Vowpal Wabbit (VW) является одной из наиболее широко используемых библиотек в индустрии. Её отличает высокая скорость работы и поддержка большого количества различных режимов обучения. Особый интерес для больших и высокоразмерных данных представляет онлайн-обучение  – самая сильная сторона библиотеки. 
# Также реализовано хэширование признаков, и Vowpal Wabbit отлично подходит для работы с текстовыми данными.
# 
# Основным интерфейсом для работы с VW является shell.

# In[22]:


get_ipython().system('vw --help')


# Vowpal Wabbit считывает данные из файла или стандартного ввода (stdin) в формате, который имеет следующий вид:
# 
# `[Label] [Importance] [Tag]|Namespace Features |Namespace Features ... |Namespace Features`
# 
# `Namespace=String[:Value]`
# 
# `Features=(String[:Value] )*`
# 
# где [] обозначает необязательные элементы, а (...)\* означает повтор неопределенное число раз. 
# 
# - **Label** является числом, "правильным" ответом. В случае классификации обычно принимает значение 1/-1, а в случае регрессии некоторое вещественное число
# - **Importance** является числом и отвечает за вес примера при обучении. Это позволяет бороться с проблемой несбалансированных данных, изученной нами ранее
# - **Tag** является некоторой строкой без пробелов и отвечает за некоторое "название" примера, которое сохраняется при предсказании ответа. Для того, чтобы отделить Tag от Importance лучше начинать Tag с символа '.
# - **Namespace** служит для создания отдельных пространств признаков. В аргументах Namespace именуются по первой букве, это нужно учитывать при выборе их названий
# - **Features** являются непосредственно признаками объекта внутри **Namespace**. Признаки по умолчанию имеют вес 1.0, но его можно переопределить, к примеру feature:0.1. 
# 
# 
# К примеру, под такой формат подходит следующая строка:
# 
# ```
# 1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park
# ```
# 
# 
# чтобы убедиться в этом, запустим vw с этим обучающим примером:

# In[23]:


get_ipython().system(" echo '1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park' | vw")


# VW является прекрасным инструментом для работы с текстовыми данными. Убедимся в этом с помощью выборки 20newsgroups, содержащей письма из 20 различных тематических рассылок.
# 
# 
# ## Новости. Бинарная классификация

# In[24]:


newsgroups = fetch_20newsgroups('../../data/news_data')


# In[25]:


newsgroups['target_names']


# Рассмотрим первый текстовый документ этой коллекции:

# In[26]:


text = newsgroups['data'][0]
target = newsgroups['target_names'][newsgroups['target'][0]]

print('-----')
print(target)
print('-----')
print(text.strip())
print('----')


# Приведем данные к формату Vowpal Wabbit, при этом оставляя только слова не короче 3 символов. Здесь мы не выполняем многие важные в анализе текстов процедуры (стемминг и лемматизацию), но, как увидим, задача и так будет решаться хорошо.

# In[27]:


def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'

to_vw_format(text, 1 if target == 'rec.autos' else -1)


# Разобьем выборку на обучающую и тестовую и запишем в файл преобразованные таким образом документы. Будем считать документ положительным, если он относится к рассылке про автомобили **rec.autos**. Так мы построим модель, отличающую письма про автомобили от остальных: 

# In[28]:


all_documents = newsgroups['data']
all_targets = [1 if newsgroups['target_names'][target] == 'rec.autos' 
               else -1 for target in newsgroups['target']]


# In[29]:


train_documents, test_documents, train_labels, test_labels =     train_test_split(all_documents, all_targets, random_state=7)
    
with open('../../data/news_data/20news_train.vw', 'w') as vw_train_data:
    for text, target in zip(train_documents, train_labels):
        vw_train_data.write(to_vw_format(text, target))
with open('../../data/news_data/20news_test.vw', 'w') as vw_test_data:
    for text in test_documents:
        vw_test_data.write(to_vw_format(text))


# Запустим Vowpal Wabbit на сформированном файле. Мы решаем задачу классификации, поэтому зададим функцию потерь в значение hinge (линейный SVM). Построенную модель мы сохраним в соответствующий файл `20news_model.vw`:

# In[30]:


get_ipython().system('vw -d ../../data/news_data/20news_train.vw   --loss_function hinge -f ../../data/news_data/20news_model.vw')


# Модель обучена. VW выводит достаточно много полезной информации по ходу обучения (тем не менее, ее можно погасить, если задать параметр --quiet). Подробно вывод диагностическйой информации разобран в документации VW на GitHub – [тут](https://github.com/JohnLangford/vowpal_wabbit/wiki/Tutorial#vws-diagnostic-information). Обратите внимание, что average loss снижался по ходу выполнения итераций. Для вычисления функции потерь VW использует еще не просмотренные примеры, поэтому, как правило, эта оценка является корректной. Применим обученную модель на тестовой выборке, сохраняя предсказания в файл с помощью опции -p: 

# In[31]:


get_ipython().system('vw -i ../../data/news_data/20news_model.vw -t -d ../../data/news_data/20news_test.vw -p ../../data/news_data/20news_test_predictions.txt')


# Загрузим полученные предсказания, вычислим AUC и отобразим ROC-кривую:

# In[32]:


with open('../../data/news_data/20news_test_predictions.txt') as pred_file:
    test_prediction = [float(label) 
                             for label in pred_file.readlines()]

auc = roc_auc_score(test_labels, test_prediction)
roc_curve = roc_curve(test_labels, test_prediction)

with plt.xkcd():
    plt.plot(roc_curve[0], roc_curve[1]);
    plt.plot([0,1], [0,1])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('test AUC = %f' % (auc)); plt.axis([-0.05,1.05,-0.05,1.05]);


# Полученное значения AUC говорит о высоком качестве классификации.

# # Новости. Многоклассовая классификация

# Используем ту же выборку, что в прошлой части, но решаем задачу многоклассовой классификации. Тут `Vowpal Wabbit` слегка капризничает – он любит, чтоб метки классов были распределены от 1 до K, где K – число классов в задаче классификации (в нашем случае – 20). Поэтому придется применить LabelEncoder, да еще и +1 потом добавить (`LabelEncoder` переводит метки в диапозон от 0 до K-1).

# In[33]:


all_documents = newsgroups['data']
topic_encoder = LabelEncoder()
all_targets_mult = topic_encoder.fit_transform(newsgroups['target']) + 1


# **Выборки будут те же, а метки поменяются, train_labels_mult и test_labels_mult – векторы меток от 1 до 20.**

# In[34]:


train_documents, test_documents, train_labels_mult, test_labels_mult =     train_test_split(all_documents, all_targets_mult, random_state=7)


# In[35]:


with open('../../data/news_data/20news_train_mult.vw', 'w') as vw_train_data:
    for text, target in zip(train_documents, train_labels_mult):
        vw_train_data.write(to_vw_format(text, target))
with open('../../data/news_data/20news_test_mult.vw', 'w') as vw_test_data:
    for text in test_documents:
        vw_test_data.write(to_vw_format(text))


# Обучим Vowpal Wabbit в режиме многоклассовой классификации, передав параметр `oaa` (от "one against all"), равный числу классов. Также перечислим параметры, которые можно понастраивать, и от которых качество модели может довольно значительно зависеть (более полно – в официальном [тьюториале](https://github.com/JohnLangford/vowpal_wabbit/wiki/Tutorial) по Vowpal Wabbit):
#  - темп обучения (-l, по умолчанию 0.5) – коэффициент перед изменением весов модели при каждом изменении
#  - степень убывания темпа обучения (--power_t, по умолчанию 0.5) – на практике проверено, что если темп обучения уменьшается при увеличении числа итераций стохастического градиентного спуска, то минимум функции находится лучше 
#  - функция потерь (--loss_function) – от нее, по сути, зависит обучаемый алгоритм. Про функции потерь в [документации](https://github.com/JohnLangford/vowpal_wabbit/wiki/Loss-functions)
#  - регуляризация (-l1) – тут надо обратить внимание на то, что в VW регуляризация считается для каждого объекта, поэтому коэффициенты регуляризации обычно берутся малыми, около $10^{-20}.$
#  
#  Дополнительно можно попробовать автоматическую настройку параметров Vowpal Wabbit с Hyperopt. Пока это работает только с Python 2. [Статья](https://habrahabr.ru/company/dca/blog/272697/) на Хабре.

# In[36]:


get_ipython().run_cell_magic('time', '', '!vw --oaa 20 ../../data/news_data/20news_train_mult.vw \\\n-f ../../data/news_data/20news_model_mult.vw --loss_function=hinge')


# In[37]:


get_ipython().run_cell_magic('time', '', '!vw -i ../../data/news_data/20news_model_mult.vw -t \\\n-d ../../data/news_data/20news_test_mult.vw \\\n-p ../../data/news_data/20news_test_predictions_mult.txt')


# In[38]:


with open('../../data/news_data/20news_test_predictions_mult.txt') as pred_file:
    test_prediction_mult = [float(label) 
                            for label in pred_file.readlines()]


# In[39]:


accuracy_score(test_labels_mult, test_prediction_mult)


# В качестве примера анализа резльтатов, посмотрим, с какими темами классификатор путает атеизм.

# In[40]:


M = confusion_matrix(test_labels_mult, test_prediction_mult)
for i in np.where(M[0,:] > 0)[0][1:]:
    print(newsgroups['target_names'][i], M[0,i], )


# # Рецензии к фильмам IMDB

# В этой части мы будем заниматься бинарной классификацией отзывов к фильмам, публикованным на сайте IMDB. Обратите внимание, насколько быстро будет работать Vowpal Wabbit.
# 
# Используем функцию `load_files` из `sklearn.datasets` для загрузки отзывов по фильмам [отсюда](https://drive.google.com/file/d/1xq4l5c0JrcxJdyBwJWvy0u9Ad_pvkJ1l/view?usp=sharing). Скачайте данные и укажите свой путь к каталогу `imdb_reviews` (в нем должны быть каталоги *train* и *test*). Разархивирование может занять несколько минут – там 100 тыс. файлов. В обучающей и тестовой выборках по 12500 тысяч хороших и плохих отзывов к фильмам. Отделим данные (собственно, тексты) от меток.

# In[41]:


# поменяйте на свой путь
path_to_movies = '/Users/y.kashnitsky/Yandex.Disk.localized/ML/data/imdb_reviews/'
reviews_train = load_files(os.path.join(path_to_movies, 'train'))
text_train, y_train = reviews_train.data, reviews_train.target


# In[42]:


print("Number of documents in training data: %d" % len(text_train))
print(np.bincount(y_train))


# То же самое с тестовой выборкой.

# In[43]:


reviews_test = load_files(os.path.join(path_to_movies, 'test'))
text_test, y_test = reviews_test.data, reviews_train.target
print("Number of documents in test data: %d" % len(text_test))
print(np.bincount(y_test))


# Примеры отзывов и соответствующих меток

# In[44]:


text_train[0]


# In[45]:


y_train[0] # хороший отзыв


# In[46]:


text_train[1]


# In[47]:


y_train[1] # плохой отзыв


# Будем использовать ранее написанную функцию `to_vw_format`.

# In[48]:


to_vw_format(str(text_train[1]), 1 if y_train[0] == 1 else -1)


# Подготовим обучающую (`movie_reviews_train.vw`), отложенную (`movie_reviews_valid.vw`) и тестовую (`movie_reviews_test.vw`) выборки для Vowpal Wabbit. 70% исходной обучаюшей выборки оставим под обучение, 30% – под отложенную выборку.

# In[49]:


train_share = int(0.7 * len(text_train))
train, valid = text_train[:train_share], text_train[train_share:]
train_labels, valid_labels = y_train[:train_share], y_train[train_share:]


# In[50]:


len(train_labels), len(valid_labels)


# In[51]:


with open('../../data/movie_reviews_train.vw', 'w') as vw_train_data:
    for text, target in zip(train, train_labels):
        vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
with open('../../data/movie_reviews_valid.vw', 'w') as vw_train_data:
    for text, target in zip(valid, valid_labels):
        vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
with open('../../data/movie_reviews_test.vw', 'w') as vw_test_data:
    for text in text_test:
        vw_test_data.write(to_vw_format(str(text)))


# In[52]:


get_ipython().system('head -2 ../../data/movie_reviews_train.vw')


# In[53]:


get_ipython().system('head -2 ../../data/movie_reviews_valid.vw')


# In[54]:


get_ipython().system('head -2 ../../data/movie_reviews_test.vw')


# **Обучим модель Vowpal Wabbit со следующими агрументами:**
# 
#  - -d, путь к обучающей выборке (соотв. файл .vw )
#  - --loss_function – hinge (хотя можно и поэкспериментировать с другими)
#  - -f – путь к файлу, в который запишется модель (можно тоже в формате .vw)

# In[55]:


get_ipython().system('vw -d ../../data/movie_reviews_train.vw --loss_function hinge -f movie_reviews_model.vw --quiet')


# Сделаем прогноз для отложенной выборки с помощью обученной модели Vowpal Wabbit, передав следующие аргументы:
#  - -i –путь к обученной модели (соотв. файл .vw)
#  - -t -d – путь к отложенной выборке (соотв. файл .vw)
#  - -p – путь к txt-файлу, куда запишутся прогнозы

# In[56]:


get_ipython().system('vw -i movie_reviews_model.vw -t -d ../../data/movie_reviews_valid.vw -p movie_valid_pred.txt --quiet')


# Считаем прогноз из файла и посчитаем долю правильных ответов и ROC AUC. Учтем, что VW выводит оценки вероятности принадлежности к классу +1. Эти оценки распределены на [-1, 1], поэтому бинарным ответом алгоритма (0 или 1) будем попросту считать тот факт, что оценка получилась положительной.

# In[57]:


with open('movie_valid_pred.txt') as pred_file:
    valid_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(valid_labels, 
               [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))


# Сделаем то же самое для тестовой выборки.

# In[58]:


get_ipython().system('vw -i movie_reviews_model.vw -t -d ../../data/movie_reviews_test.vw -p movie_test_pred.txt --quiet')


# In[59]:


with open('movie_test_pred.txt') as pred_file:
    test_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(y_test, 
               [int(pred_prob > 0) for pred_prob in test_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction), 3)))


# Попробуем улучшить прогноз за счет задействования биграмм

# In[60]:


get_ipython().system('vw -d ../../data/movie_reviews_train.vw --loss_function hinge --ngram 2 -f movie_reviews_model2.vw --quiet')


# In[61]:


get_ipython().system('vw -i movie_reviews_model2.vw -t -d ../../data/movie_reviews_valid.vw -p movie_valid_pred2.txt --quiet')


# In[62]:


with open('movie_valid_pred2.txt') as pred_file:
    valid_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(valid_labels, 
               [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))


# In[63]:


get_ipython().system('vw -i movie_reviews_model2.vw -t -d ../../data/movie_reviews_test.vw -p movie_test_pred2.txt --quiet')


# In[64]:


with open('movie_test_pred2.txt') as pred_file:
    test_prediction2 = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(y_test, 
               [int(pred_prob > 0) for pred_prob in test_prediction2]), 3)))
print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction2), 3)))


# Видим, что биграммы помогли повысить качество классификации.

# ## Классификация вопросов на StackOverflow 

# Теперь посмотрим, как в действительности Vowpal Wabbit справляется с большими выборками. Имеются 10 Гб вопросов со StackOverflow – [ссылка](https://cloud.mail.ru/public/3bwi/bFYHDN5S5) на данные, там аккурат 10 миллионов вопросов, и у каждого вопроса может быть несколько тегов. Данные довольно чистые, и не называйте это бигдатой даже в пабе :)
# 
# <img src='../../img/say_big_data.jpg' width=50%>
# 
# Из всех тегов выделены 10, и решается задача классификации на 10 классов: по тексту вопроса надо поставить один из 10 тегов, соответствующих 10 популярным языкам программирования. Предобработанные данные не даются, поскольку их надо получить в домашней работе.

# In[65]:


# поменяйте путь к данным
PATH_TO_DATA = '/Users/y.kashnitsky/Documents/Machine_learning/org_mlcourse_open/private/stackoverflow_hw/'


# In[66]:


get_ipython().system('du -hs $PATH_TO_DATA/stackoverflow_10mln_*.vw')


# Вот как выглядят строки, на которых будет обучаться Vowpal Wabbit. 10 означает 10 класс, далее вертикальная черта и просто текст вопроса. 

# In[67]:


get_ipython().system('head -1 $PATH_TO_DATA/stackoverflow_10mln_train.vw')


# Обучим на обучающей части выборки (3.3 Гб) модель Vowpal Wabbit со следующими аргументами: 
# - -oaa 10 – указываем, что классификация на 10 классов 
# - -d – путь к данным 
# - -f – путь к модели, которая будет построена 
# - -b 28 – используем 28 бит для хэширования, то есть признаковое пространство ограничено $2^{28}$ признаками, что в данном случае больше, чем число уникальных слов в выборке (но потом появятся би- и триграммы, и ограничение размерности признакового пространства начнет работать)
# - также указываем random seed

# In[68]:


get_ipython().run_cell_magic('time', '', '!vw --oaa 10 -d $PATH_TO_DATA/stackoverflow_10mln_train.vw \\\n-f vw_model1_10mln.vw -b 28 --random_seed 17 --quiet')


# Заметим, что модель обучилась всего за 43 секунды, для тестовой выборки прогнозы сделала еще за 15 секунд, доля правильных ответов – почти 92%. Далее качество модели можно повышать за счет нескольких проходов по выборке, задействования биграмм и настройке параметров. Это вместе с предобработкой данных и будет второй частью домашнего задания.

# In[69]:


get_ipython().run_cell_magic('time', '', '!vw -t -i vw_model1_10mln.vw \\\n-d $PATH_TO_DATA/stackoverflow_10mln_test.vw \\\n-p vw_valid_10mln_pred1.csv --random_seed 17 --quiet')


# In[70]:


import os
import numpy as np
from sklearn.metrics import accuracy_score

vw_pred = np.loadtxt('vw_valid_10mln_pred1.csv')
test_labels = np.loadtxt(os.path.join(PATH_TO_DATA, 
                                      'stackoverflow_10mln_test_labels.txt'))
accuracy_score(test_labels, vw_pred)

