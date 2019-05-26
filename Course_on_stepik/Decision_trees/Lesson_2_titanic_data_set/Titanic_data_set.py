#!/usr/bin/env python
# coding: utf-8

# In[29]:


from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 8)


# In[3]:


titanic_data = pd.read_csv('train.csv')
titanic_data.head()


# In[4]:


titanic_data.isnull().sum()


# In[6]:


X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived


# In[9]:


X = pd.get_dummies(X)
X.head()


# In[10]:


X = X.fillna({'Age': X.Age.median()})


# In[11]:


X.isnull().sum()


# In[12]:


clf = tree.DecisionTreeClassifier(criterion='entropy')


# In[13]:


clf.fit(X, y)


# In[30]:


tree.plot_tree(clf, feature_names=list(X),
               class_names=['Died', 'Survived'],
               filled=True);


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[33]:


X_train.shape, X_test.shape


# In[34]:


clf.score(X, y)


# In[35]:


clf.fit(X_train, y_train)


# In[36]:


clf.score(X_train, y_train)


# In[37]:


clf.score(X_test, y_test)


# In[42]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)


# In[43]:


clf.fit(X_train, y_train)


# In[44]:


clf.score(X_train, y_train)


# In[45]:


clf.score(X_test, y_test)


# In[58]:


max_depth_values = range(1, 100)


# In[64]:


scores_data = pd.DataFrame()


# In[65]:


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


# In[66]:


scores_data.head()


# In[70]:


scores_data_long = pd.melt(scores_data, 
                           id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score', 'cross_val_score'], 
                           var_name='set_type', 
                           value_name='score')


# In[73]:


scores_data_long.query("set_type == 'cross_val_score'").head(20)


# In[72]:


sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long);


# In[54]:


from sklearn.model_selection import cross_val_score


# In[55]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)


# In[57]:


cross_val_score(clf, X_train, y_train, cv=5).mean()


# In[74]:


best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=9)


# In[77]:


best_clf.fit(X_train, y_train)


# In[78]:


best_clf.score(X_test, y_test)


# In[83]:


import numpy as np
np.random.seed(0)


# In[84]:


df = pd.read_csv('train_iris.csv', index_col=0)
df.head()


# In[87]:


X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df.species


# In[91]:


df_test = pd.read_csv('test_iris.csv', index_col=0)
df_test.head()


# In[92]:


X_test = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y_test = df.species


# In[93]:


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


# In[94]:


scores_data.head()


# In[96]:


scores_data_long = pd.melt(scores_data, 
                           id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score', 'cross_val_score'], 
                           var_name='set_type', 
                           value_name='score')


# In[98]:


sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long);


# In[118]:


df_dogs_n_cats = pd.read_csv('dogs_n_cats.csv')
df_dogs_n_cats.head()


# In[119]:


y = df_dogs_n_cats['Вид']
X = df_dogs_n_cats[['Длина', 'Высота', 'Шерстист', 'Гавкает', 'Лазает по деревьям']]


# In[120]:


clf_dnc = tree.DecisionTreeClassifier(criterion='entropy')


# In[121]:


clf_dnc.fit(X, y)


# In[122]:


X.shape


# In[123]:


df_dnc_test = pd.read_json('dataset_209691_15.txt')


# In[124]:


df_dnc_test.head()


# In[125]:


df_dnc_test.shape


# In[135]:


n = clf_dnc.predict(df_dnc_test)


# In[136]:


nn = pd.Series(n)


# In[141]:


nn.value_counts()

