#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set(rc={'figure.figsize': (12, 6)})


# In[4]:


event_data = pd.read_csv('event_data_train.zip', compression='zip')


# In[5]:


event_data.head(10)


# In[6]:


event_data.action.unique()


# In[7]:


event_data['date'] = pd.to_datetime(event_data.timestamp, unit='s')


# In[8]:


event_data.head()


# In[9]:


event_data.dtypes


# In[10]:


event_data.date.min()


# In[11]:


event_data.date.max()


# In[12]:


event_data['day'] = event_data.date.dt.date


# In[13]:


event_data.head()


# In[14]:


event_data.groupby('day').user_id.nunique().plot();


# In[15]:


event_data[event_data.action == 'passed']     .groupby('user_id', as_index=False)     .agg({'step_id': 'count'})     .rename(columns={'step_id': 'passed_steps'}).passed_steps.hist();


# In[16]:


users_events_data = event_data.pivot_table(index='user_id', 
                       columns='action', 
                       values='step_id', 
                       aggfunc='count', 
                       fill_value=0).reset_index()


# In[17]:


users_events_data.head()


# In[18]:


submissions_data = pd.read_csv('submissions_data_train.zip', compression='zip')


# In[19]:


submissions_data.head()


# In[20]:


submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
submissions_data['day'] = submissions_data.date.dt.date


# In[21]:


users_scores = submissions_data.pivot_table(index='user_id', 
                       columns='submission_status', 
                       values='step_id', 
                       aggfunc='count', 
                       fill_value=0).reset_index()


# In[22]:


gap_data = event_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day'])     .groupby('user_id')['timestamp'].apply(list)     .apply(np.diff).values


# In[23]:


gap_data = pd.Series(np.concatenate(gap_data, axis=0))


# In[24]:


gap_data = gap_data / (24 * 60 * 60)


# In[25]:


gap_data[gap_data < 200].hist();


# In[26]:


gap_data.quantile(0.95)


# In[27]:


submissions_data.groupby('user_id', as_index=False).aggregate({'date': 'count'}).max()


# In[28]:


submissions_data.head()


# In[29]:


submissions_data_corr = submissions_data[submissions_data.submission_status == 'correct']


# In[30]:


ss = submissions_data_corr.groupby('user_id', as_index=False).aggregate({'submission_status': 'count'})


# In[31]:


event_data.tail()


# In[32]:


users_data = event_data.groupby('user_id', as_index=False)     .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})


# In[33]:


now = 1526772811
drop_out_threshold = 2592000


# In[34]:


users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold


# In[35]:


users_data.head()


# In[36]:


users_scores.head()


# In[37]:


users_data = users_data.merge(users_scores, on='user_id', how='outer')


# In[38]:


users_data = users_data.fillna(0)


# In[39]:


users_data = users_data.merge(users_events_data, on='user_id', how='outer')


# In[40]:


users_days = event_data.groupby('user_id').day.nunique().to_frame().reset_index().head()


# In[41]:


users_data = users_data.merge(users_days, on='user_id', how='outer')


# In[42]:


users_data.head()


# In[43]:


users_data.user_id.nunique()


# In[44]:


event_data.user_id.nunique()


# In[45]:


users_data['passed_course'] = users_data.passed > 170


# In[46]:


users_data.head()


# In[47]:


users_data.groupby('passed_course').count()


# In[48]:


100 * 1425 / 17809


# In[49]:


event_data.head()


# In[50]:


user_min_time = event_data.groupby('user_id', as_index=False)     .agg({'timestamp': 'min'})     .rename({'timestamp': 'min_timestamp'}, axis=1)


# In[51]:


users_data = users_data.merge(user_min_time, how='outer')


# In[52]:


event_data['user_time'] = event_data.user_id.map(str) + '_' + event_data.timestamp.map(str)


# In[53]:


event_data.shape


# In[54]:


learning_time_threshold = 3 * 24 * 60 * 60


# In[55]:


user_learning_time_threshold = user_min_time.user_id.map(str) + '_' + (user_min_time.min_timestamp + learning_time_threshold).map(str)


# In[56]:


user_min_time['user_learning_time_threshold'] = user_learning_time_threshold


# In[57]:


event_data = event_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')


# In[58]:


event_data.shape


# In[59]:


events_data_train = event_data[event_data.user_time <= event_data.user_learning_time_threshold]


# In[60]:


events_data_train.head()


# In[61]:


events_data_train.shape


# In[62]:


submissions_data.head()


# In[63]:


nn = submissions_data[submissions_data.submission_status == 'wrong'].groupby('step_id').count()


# In[64]:


nn.submission_status.sort_values().tail(1)


# In[66]:


events_data_train.groupby('user_id').day.nunique().max()


# In[67]:


submissions_data['user_time'] = submissions_data.user_id.map(str) + '_' + submissions_data.timestamp.map(str)
submissions_data = submissions_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')
submissions_data_train = submissions_data[submissions_data.user_time <= submissions_data.user_learning_time_threshold]
submissions_data_train.groupby('user_id').day.nunique().max()


# In[75]:


X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index()     .rename(columns={'day': 'days'})


# In[76]:


steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index()     .rename(columns={'step_id': 'step_tried'})


# In[77]:


X = X.merge(steps_tried, on='user_id', how='outer')


# In[78]:


X.shape


# In[79]:


X.head()


# In[81]:


X = X.merge(submissions_data_train.pivot_table(index='user_id',
                                   columns='submission_status',
                                   values='step_id',
                                   aggfunc='count',
                                   fill_value=0).reset_index())


# In[82]:


X.head()


# In[84]:


X.shape


# In[85]:


X['correct_ratio'] = X.correct / (X.correct + X.wrong)


# In[86]:


X.head()


# In[88]:


X = X.merge(events_data_train.pivot_table(index='user_id',
                                   columns='action',
                                   values='step_id',
                                   aggfunc='count',
                                   fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')


# In[89]:


X.shape


# In[91]:


X = X.fillna(0)


# In[94]:


X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']], how='outer')


# In[96]:


X.head()


# In[97]:


X = X[~((X.is_gone_user == False) & (X.passed_course == False))]


# In[98]:


X.head()


# In[99]:


X.groupby(['passed_course', 'is_gone_user']).user_id.count()


# In[109]:


y = X.passed_course


# In[110]:


y = y.map(int)


# In[101]:


X = X.drop(['passed_course', 'is_gone_user'], axis=1)


# In[102]:


X.head()


# In[105]:


X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1)


# In[106]:


X.head()


# In[125]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


# In[117]:


clf = DecisionTreeClassifier()


# In[118]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[119]:


clf.fit(X_train, y_train)


# In[121]:


y_pred = clf.predict(X_test)


# In[124]:


roc_auc_score(y_test, y_pred)


# In[127]:


parametrs = {'max_depth': range(1, 11), 'min_samples_split': range(2, 11), 'min_samples_leaf': range(1, 11)}
search = GridSearchCV(clf, parametrs, cv=5)
search.fit(X_train, y_train)


# In[128]:


search.estimator


# In[129]:


best_tree = search.estimator
best_tree.fit(X_train, y_train)
predictions = best_tree.predict(X_test)


# In[130]:


roc_auc_score(y_test, predictions)

