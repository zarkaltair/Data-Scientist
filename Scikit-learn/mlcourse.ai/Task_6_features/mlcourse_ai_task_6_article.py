#!/usr/bin/env python
# coding: utf-8

# In[1]:


# перед началом работы не забудьте скачать файл train.json.zip с Kaggle и разархивировать его
import json
import pandas as pd

# сразу загрузим датасет от Renthop
with open('train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)


# In[2]:


from functools import reduce
import numpy as np

texts = [['i', 'have', 'a', 'cat'],
         ['he', 'have', 'a', 'dog'],
         ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(reduce(lambda x, y: x + y, texts))))

def vectorize(text):
    vector = np.zeros(len(dictionary))
    for i, word in dictionary:
        num = 0
        for w in text:
            if w == word:
                num += 1
        if num:
            vector[i] = num
    return vector

for t in texts:
    print(vectorize(t))


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,1)) 
vect.fit_transform(['no i have cows', 'i have no cows']).toarray()


# In[4]:


vect.vocabulary_


# In[5]:


vect = CountVectorizer(ngram_range=(1,2)) 
vect.fit_transform(['no i have cows', 'i have no cows']).toarray()


# In[6]:


vect.vocabulary_


# In[7]:


from scipy.spatial.distance import euclidean

vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb') 
n1, n2, n3, n4 = vect.fit_transform(['иванов', 'петров', 'петренко', 'смит']).toarray()
euclidean(n1, n2)


# In[8]:


euclidean(n2, n3)


# In[9]:


euclidean(n3, n4)


# In[ ]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from scipy.misc import face
import numpy as np

resnet_settings = {'include_top': False, 'weights': 'imagenet'}
resnet = ResNet50(**resnet_settings)

img = image.array_to_img(face())
# какой милый енот! 
img = img.resize((224, 224))
# в реальной жизни может понадобиться внимательнее относиться к ресайзу
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# нужно дополнительное измерение, т.к. модель рассчитана на работу с массивом изображений

features = resnet.predict(x)


# In[12]:


import pytesseract
from PIL import Image
import requests
from io import BytesIO

img = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
# просто случайная картинка из поиска 

img = requests.get(img)
img = Image.open(BytesIO(img.content))
text = pytesseract.image_to_string(img)
text


# In[13]:


# на этот раз возьмем картинку от Renthop
img = requests.get('https://photos.renthop.com/2/8393298_6acaf11f030217d05f3a5604b9a2f70f.jpg')
img = Image.open(BytesIO(img.content))
pytesseract.image_to_string(img)


# In[15]:


import reverse_geocoder as revgc

revgc.search((df.latitude, df.longitude))


# In[16]:


df['dow'] = df['created'].apply(lambda x: x.date().weekday())
df['is_weekend'] = df['created'].apply(lambda x: 1 if x.date().weekday() in (5, 6) else 0)


# In[17]:


def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)


# In[18]:


from scipy.spatial import distance

euclidean(make_harmonic_features(23), make_harmonic_features(1))


# In[19]:


euclidean(make_harmonic_features(9), make_harmonic_features(11))


# In[20]:


euclidean(make_harmonic_features(9), make_harmonic_features(21))


# In[24]:


ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'

import user_agents

ua = user_agents.parse(ua) 


# In[25]:


ua.is_bot


# In[26]:


ua.is_mobile


# In[27]:


ua.is_pc


# In[28]:


ua.os.family


# In[29]:


ua.os.version


# In[30]:


ua.browser.family


# In[31]:


ua.os.version


# In[32]:


ua.browser.version


# In[33]:


from sklearn.preprocessing import StandardScaler  
from scipy.stats import beta
from scipy.stats import shapiro

data = beta(1, 10).rvs(1000).reshape(-1, 1)
shapiro(data)


# In[34]:


shapiro(StandardScaler().fit_transform(data))


# In[35]:


data = np.array([1, 1, 0, -1, 2, 1, 2, 3, -2, 4, 100]).reshape(-1, 1).astype(np.float64)
StandardScaler().fit_transform(data)


# In[37]:


(data - data.mean()) / data.std()


# In[38]:


from sklearn.preprocessing import MinMaxScaler

MinMaxScaler().fit_transform(data)


# In[39]:


(data - data.min()) / (data.max() - data.min())


# In[40]:


from scipy.stats import lognorm

data = lognorm(s=1).rvs(1000)
shapiro(data)


# In[41]:


shapiro(np.log(data))


# In[43]:


import statsmodels.api as sm

# возьмем признак price из датасета Renthop и пофильтруем руками совсем экстремальные значения для наглядности
price = df.price[(df.price <= 20000) & (df.price > 500)]
price_log = np.log(price)

price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
# много телодвижений, чтобы sklearn не сыпал warning-ами

price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std()).savefig('qq_price_log.png')
sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_mm.std()).savefig('qq_price_mm.png')
sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std()).savefig('qq_price_z.png')


# In[46]:


from demo import get_data

x_data, y_data = get_data()
x_data.head(5)


# In[47]:


x_data = x_data.values

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

cross_val_score(LogisticRegression(), x_data, y_data, scoring='neg_log_loss').mean()
# кажется, что-то пошло не так! вообще-то стоит разобраться, в чем проблема


# In[ ]:


from sklearn.preprocessing import StandardScaler

cross_val_score(LogisticRegression(), StandardScaler().fit_transform(x_data), y_data, scoring='neg_log_loss').mean()
# ого! действительно помогает!


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

cross_val_score(LogisticRegression(), MinMaxScaler().fit_transform(x_data), y_data, scoring='neg_log_loss').mean()
# a на этот раз – нет :( 


# In[48]:


rooms = df["bedrooms"].apply(lambda x: max(x, .5))
# избегаем деления на ноль; .5 выбран более или менее произвольно
df["price_per_bedroom"] = df["price"] / rooms


# In[49]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

x_data_generated, y_data_generated = make_classification()
x_data_generated.shape


# In[50]:


VarianceThreshold(.7).fit_transform(x_data_generated).shape


# In[51]:


VarianceThreshold(.8).fit_transform(x_data_generated).shape


# In[52]:


VarianceThreshold(.9).fit_transform(x_data_generated).shape


# In[57]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

x_data_kbest = SelectKBest(f_classif, k=5).fit_transform(x_data_generated, y_data_generated)
x_data_varth = VarianceThreshold(.9).fit_transform(x_data_generated)

cross_val_score(LogisticRegression(solver='lbfgs'), x_data_generated, y_data_generated, cv=5, scoring='neg_log_loss').mean()


# In[58]:


cross_val_score(LogisticRegression(solver='lbfgs'), x_data_kbest, y_data_generated, cv=5, scoring='neg_log_loss').mean()


# In[60]:


cross_val_score(LogisticRegression(solver='lbfgs'), x_data_varth, y_data_generated, cv=5, scoring='neg_log_loss').mean()


# In[65]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

x_data_generated, y_data_generated = make_classification()

pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier(n_estimators=100)),
                     LogisticRegression(solver='lbfgs'))

lr = LogisticRegression(solver='lbfgs')
rf = RandomForestClassifier(n_estimators=100)

print(cross_val_score(lr, x_data_generated, y_data_generated, cv=5, scoring='neg_log_loss').mean())
print(cross_val_score(rf, x_data_generated, y_data_generated, cv=5, scoring='neg_log_loss').mean())
print(cross_val_score(pipe, x_data_generated, y_data_generated, cv=5, scoring='neg_log_loss').mean())


# In[66]:


x_data, y_data = get_data()
x_data = x_data.values

pipe1 = make_pipeline(StandardScaler(),
                      SelectFromModel(estimator=RandomForestClassifier(n_estimators=100)),
                      LogisticRegression(solver='lbfgs'))

pipe2 = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='lbfgs'))

rf = RandomForestClassifier(n_estimators=100)

print('LR + selection: ', cross_val_score(pipe1, x_data, y_data, cv=5, scoring='neg_log_loss').mean())
print('LR: ', cross_val_score(pipe2, x_data, y_data, cv=5, scoring='neg_log_loss').mean())
print('RF: ', cross_val_score(rf, x_data, y_data, cv=5, scoring='neg_log_loss').mean())


# In[71]:


from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(LogisticRegression(solver='lbfgs'), scoring='neg_log_loss', verbose=2, k_features=3, forward=False, n_jobs=-1)
selector.fit(x_data_scaled, y_data)
selector.fit(x_data_scaled, y_data)


# In[ ]:





# In[ ]:




