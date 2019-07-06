#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## sklearn.metrics

# документация: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# In[2]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Генерация датасетов

# In[3]:


clf_data, clf_target = datasets.make_classification(n_features=2, n_informative=2, n_classes=2, 
                                                    n_redundant=0, n_clusters_per_class=1, 
                                                    random_state=7)


# In[4]:


reg_data, reg_target = datasets.make_regression(n_features=2, n_informative=1, n_targets=1, 
                                                noise=5, random_state=7)


# In[9]:


colors = ListedColormap(['red', 'blue'])
plt.scatter([i[0] for i in reg_data], [i[1] for i in reg_data], c=clf_target, cmap=colors);


# In[10]:


plt.scatter([i[0] for i in reg_data], reg_target, color='r')
plt.scatter([i[1] for i in reg_data], reg_target, color='b');


# In[11]:


clf_train_data, clf_test_data, clf_train_labels, clf_test_labels = train_test_split(clf_data, 
                                                                                    clf_target,
                                                                                    test_size=0.3,
                                                                                    random_state=1)


# In[12]:


reg_train_data, reg_test_data, reg_train_labels, reg_test_labels = train_test_split(reg_data, 
                                                                                    reg_target,
                                                                                    test_size=0.3,
                                                                                    random_state=1)


# ### Метрики качества в задачах классификации

# #### Обучение модели классификации

# In[13]:


classifier = linear_model.SGDClassifier(loss='log', random_state=1)


# In[14]:


classifier.fit(clf_train_data, clf_train_labels)


# In[15]:


predictions = classifier.predict(clf_test_data)


# In[16]:


probability_predictions = classifier.predict_proba(clf_test_data)


# In[17]:


clf_test_labels


# In[18]:


predictions


# In[19]:


probability_predictions


# #### accuracy

# In[20]:


sum([1. if pair[0] == pair[1] else 0. for pair in zip(clf_test_labels, predictions)])/len(clf_test_labels)


# In[21]:


metrics.accuracy_score(clf_test_labels, predictions)


# #### confusion matrix

# In[22]:


matrix = metrics.confusion_matrix(clf_test_labels, predictions)
matrix


# In[23]:


sum([1 if pair[0] == pair[1] else 0 for pair in zip(clf_test_labels, predictions)])


# In[24]:


matrix.diagonal().sum()


# #### precision 

# In[25]:


metrics.precision_score(clf_test_labels, predictions, pos_label=0)


# In[26]:


metrics.precision_score(clf_test_labels, predictions)


# #### recall

# In[27]:


metrics.recall_score(clf_test_labels, predictions, pos_label=0)


# In[28]:


metrics.recall_score(clf_test_labels, predictions)


# #### f1

# In[29]:


metrics.f1_score(clf_test_labels, predictions, pos_label=0)


# In[30]:


metrics.f1_score(clf_test_labels, predictions)


# #### classification report

# In[31]:


metrics.classification_report(clf_test_labels, predictions)


# #### ROC curve

# In[32]:


fpr, tpr, _ = metrics.roc_curve(clf_test_labels, probability_predictions[:,1])


# In[33]:


plt.plot(fpr, tpr, label='linear model')
plt.plot([0, 1], [0, 1], '--', color='grey', label='random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right");


# #### ROC AUC

# In[34]:


metrics.roc_auc_score(clf_test_labels, predictions)


# In[35]:


metrics.roc_auc_score(clf_test_labels, probability_predictions[:,1])


# #### PR AUC

# In[36]:


metrics.average_precision_score(clf_test_labels, predictions)


# #### log_loss

# In[37]:


metrics.log_loss(clf_test_labels, probability_predictions[:,1])


# ### Метрики качества в задачах регрессии

# #### Обучение регрессионной модели 

# In[41]:


regressor = linear_model.SGDRegressor(random_state=1, max_iter=1000)


# In[42]:


regressor.fit(reg_train_data, reg_train_labels)


# In[43]:


reg_predictions = regressor.predict(reg_test_data)


# In[44]:


reg_test_labels


# In[45]:


reg_predictions


# #### mean absolute error

# In[46]:


metrics.mean_absolute_error(reg_test_labels, reg_predictions)


# #### mean squared error

# In[47]:


metrics.mean_squared_error(reg_test_labels, reg_predictions)


# #### root mean squared error

# In[49]:


import numpy as np


# In[50]:


np.sqrt(metrics.mean_squared_error(reg_test_labels, reg_predictions))


# #### r2 score

# In[51]:


metrics.r2_score(reg_test_labels, reg_predictions)

