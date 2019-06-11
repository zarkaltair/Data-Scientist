#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# ## Task #1

# In[2]:


words = []
file = open('sentences.txt')
for line in file:
    for word in re.split('[^a-z]', line.lower()):
        words.append(word)


# In[16]:


arr = list(dict.fromkeys(words))


# In[18]:


arr.remove('')


# In[20]:


len(arr)


# In[31]:


dict_words = dict(zip(arr, [i for i in range(len(arr))]))


# In[56]:


file = open('sentences.txt')
arr_entire = []
for line in file:
    list_words = re.split('[^a-z]', line.lower())
    list_words = [i for i in list_words if i != '']
    for key in dict_words:
        arr_entire.append(list_words.count(key))


# In[58]:


matrix = np.array(arr_entire).reshape(22, 254)


# In[59]:


matrix.shape


# In[60]:


from scipy.spatial.distance import cosine


# In[61]:


cosine(matrix[0], matrix[0])


# In[62]:


arr_dist = []
for i in range(1, 22):
    arr_dist.append(cosine(matrix[0], matrix[i]))


# In[63]:


arr_dist


# ## Task #2

# In[217]:


from scipy.linalg import solve


# In[243]:


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


# In[244]:


x = np.linspace(1, 16)


# In[246]:


plt.plot(x, f(x));


# In[250]:


a1 = np.array([[1, 1], [1, 15]])
b1 = np.array([f(1), f(15)])
s1 = solve(a, b)


# In[251]:


def f1(x):
    return s1[0] + s1[1] * x


# In[252]:


plt.plot(x, func(x), x, f1(x));


# In[253]:


a2 = np.array([[1, 1, 1], [1, 8, 8 ** 2], [1, 15, 15 ** 2]])
b2 = np.array([f(1), f(8), f(15)])
s2 = solve(a2, b2)


# In[256]:


def f2(x):
    return s2[0] + s2[1] * x + s2[2] * x ** 2


# In[257]:


plt.plot(x, f(x), x, f1(x), x, f2(x));


# In[263]:


a3 = np.array([[1, 1, 1, 1], [1, 4, 4 ** 2, 4 ** 3], [1, 10, 10 ** 2, 10 ** 3], [1, 15, 15 ** 2, 15 ** 3]])
b3 = np.array([f(1), f(4), f(10), f(15)])
s3 = solve(a3, b3)


# In[264]:


def f3(x):
    return s3[0] + s3[1] * x + s3[2] * x ** 2 + s3[3] * x ** 3


# In[265]:


plt.plot(x, f(x), x, f1(x), x, f2(x), x, f3(x));


# In[266]:


s3

