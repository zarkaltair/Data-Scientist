#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## sklearn.datasets

# документация: http://scikit-learn.org/stable/datasets/

# In[1]:


from sklearn import datasets


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Генерация выборок

# **Способы генерации данных:** 
# * make_classification
# * make_regression
# * make_circles
# * make_checkerboard
# * etc

# #### datasets.make_circles

# In[3]:


circles = datasets.make_circles()


# In[62]:


print('features:\n{}\n'.format(circles[0][:10]))
print("target:\n{}".format(circles[1][:10]))


# In[108]:


from matplotlib.colors import ListedColormap


# In[115]:


colors = ListedColormap(['red', 'yellow'])
x = [i[0] for i in circles[0]]
y = [i[1] for i in circles[0]]

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=circles[1], cmap=colors);


# In[97]:


def plot_2d_dataset(data, colors):
    plt.figure(figsize=(8, 8))
    x = [i[0] for i in circles[0]]
    y = [i[1] for i in circles[0]]
    plt.scatter(x, y, c=data[1], cmap=colors)


# In[116]:


noisy_circles = datasets.make_circles(noise=0.5)


# In[117]:


plot_2d_dataset(noisy_circles, colors)


# #### datasets.make_classification

# In[100]:


simple_classification_problem = datasets.make_classification(n_features = 2, n_informative = 1, 
                                                             n_redundant = 1, n_clusters_per_class = 1,
                                                             random_state = 1 )


# In[101]:


plot_2d_dataset(simple_classification_problem, colors)


# In[102]:


classification_problem = datasets.make_classification(n_features = 2, n_informative = 2, n_classes = 4, 
                                                      n_redundant = 0, n_clusters_per_class = 1, random_state = 1)

colors = ListedColormap(['red', 'blue', 'green', 'yellow'])


# In[103]:


plot_2d_dataset(classification_problem, colors)


# ### "Игрушечные" наборы данных

# **Наборы данных:** 
# * load_iris 
# * load_boston
# * load_diabetes
# * load_digits
# * load_linnerud
# * etc

# #### datasets.load_iris

# In[21]:


iris = datasets.load_iris()


# In[32]:


iris.keys()


# In[34]:


print(iris.DESCR)


# In[35]:


print("feature names: {}".format(iris.feature_names))
print("target names: {names}".format(names = iris.target_names))


# In[36]:


iris.data[:10]


# In[37]:


iris.target


# ### Визуализация выбокри

# In[38]:


from pandas import DataFrame


# In[39]:


iris_frame = DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame['target'] = iris.target


# In[40]:


iris_frame.head()


# In[41]:


iris_frame.target = iris_frame.target.apply(lambda x : iris.target_names[x])


# In[42]:


iris_frame.head()


# In[44]:


iris_frame[iris_frame.target == 'setosa'].hist('sepal length (cm)');


# In[46]:


plt.figure(figsize=(20, 24))

plot_number = 0
for feature_name in iris['feature_names']:
    for target_name in iris['target_names']:
        plot_number += 1
        plt.subplot(4, 3, plot_number)
        plt.hist(iris_frame[iris_frame.target == target_name][feature_name])
        plt.title(target_name)
        plt.xlabel('cm')
        plt.ylabel(feature_name[:-4])


# ### Бонус: библиотека seaborn

# In[47]:


import seaborn as sns


# In[49]:


sns.pairplot(iris_frame, hue='target');


# In[53]:


sns.set(font_scale = 1.3)
data = sns.load_dataset("iris")
sns.pairplot(data, hue="species");


# #### **Если Вас заинтересовала библиотека seaborn:**
# * установка: https://stanford.edu/~mwaskom/software/seaborn/installing.html
# * установка c помощью анаконды: https://anaconda.org/anaconda/seaborn
# * руководство: https://stanford.edu/~mwaskom/software/seaborn/tutorial.html
# * примеры: https://stanford.edu/~mwaskom/software/seaborn/examples/
