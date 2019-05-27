#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn import tree
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 8)


# In[2]:


titanic_data = pd.read_csv('train.csv')
titanic_data.head()


# In[3]:


titanic_data.isnull().sum()


# In[4]:


X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived


# In[5]:


X = pd.get_dummies(X)
X.head()


# In[6]:


X = X.fillna({'Age': X.Age.median()})


# In[7]:


X.isnull().sum()


# In[8]:


clf = tree.DecisionTreeClassifier(criterion='entropy')


# In[9]:


clf.fit(X, y)


# In[10]:


tree.plot_tree(clf, feature_names=list(X),
               class_names=['Died', 'Survived'],
               filled=True);


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[13]:


X_train.shape, X_test.shape


# In[14]:


clf.score(X, y)


# In[15]:


clf.fit(X_train, y_train)


# In[16]:


clf.score(X_train, y_train)


# In[17]:


clf.score(X_test, y_test)


# In[18]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)


# In[19]:


clf.fit(X_train, y_train)


# In[20]:


clf.score(X_train, y_train)


# In[21]:


clf.score(X_test, y_test)


# In[22]:


max_depth_values = range(1, 100)


# In[23]:


scores_data = pd.DataFrame()


# In[24]:


for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 
                                    'train_score': [train_score], 
                                    'test_score': [test_score], 
                                    'cross_val_score': [mean_cross_val_score]})
    
    scores_data = scores_data.append(temp_score_data)


# In[25]:


scores_data.head()


# In[26]:


scores_data_long = pd.melt(scores_data, 
                           id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score', 'cross_val_score'], 
                           var_name='set_type', 
                           value_name='score')


# In[27]:


scores_data_long.query("set_type == 'cross_val_score'").head(20)


# In[28]:


sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long);


# In[29]:


from sklearn.model_selection import cross_val_score


# In[30]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)


# In[31]:


cross_val_score(clf, X_train, y_train, cv=5).mean()


# In[32]:


best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=9)


# In[33]:


best_clf.fit(X_train, y_train)


# In[34]:


best_clf.score(X_test, y_test)


# In[35]:


import numpy as np
np.random.seed(0)


# In[36]:


df = pd.read_csv('train_iris.csv', index_col=0)
df.head()


# In[37]:


X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df.species


# In[38]:


df_test = pd.read_csv('test_iris.csv', index_col=0)
df_test.head()


# In[39]:


X_test = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y_test = df.species


# In[40]:


for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X, y)
    train_score = clf.score(X, y)
    test_score = clf.score(X_test, y_test)
    
    mean_cross_val_score = cross_val_score(clf, X, y, cv=5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 
                                    'train_score': [train_score], 
                                    'test_score': [test_score], 
                                    'cross_val_score': [mean_cross_val_score]})
    
    scores_data = scores_data.append(temp_score_data)


# In[41]:


scores_data.head()


# In[42]:


scores_data_long = pd.melt(scores_data, 
                           id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score', 'cross_val_score'], 
                           var_name='set_type', 
                           value_name='score')


# In[43]:


sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long);


# In[44]:


df_dogs_n_cats = pd.read_csv('dogs_n_cats.csv')
df_dogs_n_cats.head()


# In[45]:


y = df_dogs_n_cats['Вид']
X = df_dogs_n_cats[['Длина', 'Высота', 'Шерстист', 'Гавкает', 'Лазает по деревьям']]


# In[46]:


clf_dnc = tree.DecisionTreeClassifier(criterion='entropy')


# In[47]:


clf_dnc.fit(X, y)


# In[48]:


X.shape


# In[49]:


df_dnc_test = pd.read_json('dataset_209691_15.txt')


# In[50]:


df_dnc_test.head()


# In[51]:


df_dnc_test.shape


# In[52]:


n = clf_dnc.predict(df_dnc_test)


# In[53]:


nn = pd.Series(n)


# In[54]:


nn.value_counts()


# In[35]:


from sklearn.model_selection import GridSearchCV


# In[36]:


clf_grid = tree.DecisionTreeClassifier()


# In[37]:


parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}


# In[38]:


grid_search_cv_clf = GridSearchCV(clf_grid, parametrs, cv=5)


# In[39]:


grid_search_cv_clf.fit(X_train, y_train)


# In[40]:


grid_search_cv_clf.best_params_


# In[41]:


best_clf = grid_search_cv_clf.best_estimator_


# In[42]:


best_clf.score(X_test, y_test)


# In[46]:


from sklearn.metrics import precision_score, recall_score


# In[44]:


y_pred = best_clf.predict(X_test)


# In[45]:


precision_score(y_test, y_pred)


# In[47]:


recall_score(y_test, y_pred)


# In[48]:


y_predicted_prob = best_clf.predict_proba(X_test)


# In[53]:


pd.Series(y_predicted_prob[:, 1]).hist();


# In[61]:


y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)


# In[62]:


precision_score(y_test, y_pred)


# In[63]:


recall_score(y_test, y_pred)


# In[65]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show();

