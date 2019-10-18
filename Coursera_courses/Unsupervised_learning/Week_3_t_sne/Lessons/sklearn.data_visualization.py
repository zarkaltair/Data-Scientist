#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# # Визуализация данных

# In[1]:


from sklearn import datasets

import numpy as np


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ### Загрузка выборки

# In[4]:


digits  =  datasets.load_digits()


# In[5]:


digits.DESCR


# In[6]:


print('target:', digits.target[0])
print('features: \n', digits.data[0]) 
print('number of features:', len(digits.data[0]))


# ## Визуализация объектов выборки

# In[8]:


#не будет работать: Invalid dimensions for image data
plt.imshow(digits.data[0])


# In[9]:


digits.data[0].shape


# In[10]:


digits.data[0].reshape(8,8)


# In[11]:


digits.data[0].reshape(8,8).shape


# In[13]:


plt.imshow(digits.data[0].reshape(8,8));


# In[14]:


digits.keys()


# In[15]:


digits.images[0]


# In[16]:


plt.imshow(digits.images[0]);


# In[18]:


plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(digits.images[0])

plt.subplot(2, 2, 2)
plt.imshow(digits.images[0], cmap = 'hot')

plt.subplot(2, 2, 3)
plt.imshow(digits.images[0], cmap = 'gray')

plt.subplot(2, 2, 4)
plt.imshow(digits.images[0], cmap = 'gray', interpolation = 'nearest');


# In[19]:


plt.figure(figsize=(20, 8))

for plot_number, plot in enumerate(digits.images[:10]):
    plt.subplot(2, 5, plot_number + 1)
    plt.imshow(plot, cmap = 'gray')
    plt.title('digit: ' + str(digits.target[plot_number]))


# ## Уменьшение размерности

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from collections import Counter


# In[21]:


data = digits.data[:1000]
labels = digits.target[:1000]


# In[22]:


Counter(labels)


# In[23]:


plt.figure(figsize=(10, 6))
plt.bar(Counter(labels).keys(), Counter(labels).values());


# In[24]:


classifier = KNeighborsClassifier()


# In[25]:


classifier.fit(data, labels)


# In[27]:


print(classification_report(classifier.predict(data), labels))


# ### Random projection

# In[28]:


from sklearn import random_projection


# In[29]:


projection = random_projection.SparseRandomProjection(n_components=2, random_state=0)
data_2d_rp = projection.fit_transform(data)


# In[30]:


plt.figure(figsize=(10, 6))
plt.scatter(data_2d_rp[:, 0], data_2d_rp[:, 1], c=labels);


# In[31]:


classifier.fit(data_2d_rp, labels)
print(classification_report(classifier.predict(data_2d_rp), labels))


# ### PCA

# In[32]:


from sklearn.decomposition import PCA


# In[33]:


pca = PCA(n_components=2, random_state=0)
data_2d_pca = pca.fit_transform(data);


# In[34]:


plt.figure(figsize = (10, 6))
plt.scatter(data_2d_pca[:, 0], data_2d_pca[:, 1], c=labels);


# In[35]:


classifier.fit(data_2d_pca, labels)
print(classification_report(classifier.predict(data_2d_pca), labels))


# ### MDS

# In[36]:


from sklearn import manifold


# In[37]:


mds = manifold.MDS(n_components=2, n_init=1, max_iter=100)
data_2d_mds = mds.fit_transform(data)


# In[38]:


plt.figure(figsize=(10, 6))
plt.scatter(data_2d_mds[:, 0], data_2d_mds[:, 1], c=labels);


# In[39]:


classifier.fit(data_2d_mds, labels)
print(classification_report(classifier.predict(data_2d_mds), labels))


# ### t- SNE

# In[40]:


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
data_2d_tsne = tsne.fit_transform(data)


# In[41]:


plt.figure(figsize=(10, 6))
plt.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], c=labels);


# In[42]:


classifier.fit(data_2d_tsne, labels)
print(classification_report(classifier.predict(data_2d_tsne), labels))

