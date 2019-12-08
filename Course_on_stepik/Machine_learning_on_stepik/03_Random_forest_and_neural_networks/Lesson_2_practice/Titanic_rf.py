#!/usr/bin/env python
# coding: utf-8

# In[64]:


from sklearn import tree
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12, 8)


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


# In[43]:


from sklearn.metrics import precision_score, recall_score


# In[44]:


y_pred = best_clf.predict(X_test)


# In[45]:


precision_score(y_test, y_pred)


# In[46]:


recall_score(y_test, y_pred)


# In[47]:


y_predicted_prob = best_clf.predict_proba(X_test)


# In[48]:


pd.Series(y_predicted_prob[:, 1]).hist();


# In[49]:


y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)


# In[50]:


precision_score(y_test, y_pred)


# In[51]:


recall_score(y_test, y_pred)


# In[52]:


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


# In[61]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=100, min_samples_leaf=10)


# In[62]:


clf.fit(X_train, y_train)


# In[65]:


tree.plot_tree(clf, feature_names=list(X_train),
               class_names=['Died', 'Survived'],
               filled=True);


# In[66]:


from sklearn.ensemble import RandomForestClassifier


# In[67]:


clf_rf = RandomForestClassifier()


# In[68]:


parametrs = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}


# In[69]:


grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=5)


# In[70]:


grid_search_cv_clf.fit(X_train, y_train)


# In[71]:


grid_search_cv_clf.best_params_


# In[72]:


best_clf = grid_search_cv_clf.best_estimator_


# In[73]:


best_clf.score(X_test, y_test)


# In[75]:


feature_importances = best_clf.feature_importances_


# In[76]:


feature_importances_df = pd.DataFrame({'features': list(X_train),
                                       'feature_importances': feature_importances})


# In[78]:


feature_importances_df.sort_values('feature_importances', ascending=False)


# In[80]:


df = pd.read_csv('heart.csv')
df.head()


# In[81]:


rf = RandomForestClassifier(n_estimators=10, max_depth=5)


# In[82]:


y = df.target
X = df.drop('target', axis=1)


# In[84]:


rf.fit(X, y)


# In[87]:


imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8));

