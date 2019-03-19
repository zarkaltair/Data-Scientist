#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# In[7]:


reviews_train = load_files("aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target


# In[ ]:


print("Number of documents in training data: %d" % len(text_train))
print(np.bincount(y_train))


# In[ ]:


# поменяйте путь к файлу
reviews_test = load_files("aclImdb/test")
text_test, y_test = reviews_test.data, reviews_test.target
print("Number of documents in test data: %d" % len(text_test))
print(np.bincount(y_test))


# In[ ]:


cv = CountVectorizer()
cv.fit(text_train)

print(len(cv.vocabulary_)) #74849


# In[ ]:


print(cv.get_feature_names()[:50])
print(cv.get_feature_names()[50000:50050])


# In[ ]:


X_train = cv.transform(text_train)
X_test = cv.transform(text_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'logit = LogisticRegression(n_jobs=-1, random_state=7)\nlogit.fit(X_train, y_train)\nprint(round(logit.score(X_train, y_train), 3), round(logit.score(X_test, y_test), 3))')


# In[ ]:


def visualize_coefficients(classifier, feature_names, n_top_features=25):
    # get coefficients with large absolute values 
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha="right");


# In[ ]:


def plot_grid_scores(grid, param_name):
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'],
    color='green', label='train')
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'],
    color='red', label='test')
    plt.legend();


# In[ ]:


visualize_coefficients(logit, cv.get_feature_names())


# In[ ]:


from sklearn.pipeline import make_pipeline

text_pipe_logit = make_pipeline(CountVectorizer(), 
LogisticRegression(n_jobs=-1, random_state=7))

text_pipe_logit.fit(text_train, y_train)
print(text_pipe_logit.score(text_test, y_test))

from sklearn.model_selection import GridSearchCV

param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
grid_logit = GridSearchCV(text_pipe_logit, param_grid_logit, cv=3, n_jobs=-1)

grid_logit.fit(text_train, y_train)
grid_logit.best_params_, grid_logit.best_score_
plot_grid_scores(grid_logit, 'logisticregression__C')
grid_logit.score(text_test, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=17)
forest.fit(X_train, y_train)
print(round(forest.score(X_test, y_test), 3))


# In[ ]:


# порождаем данные
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)


# In[ ]:


plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired);


# In[ ]:


def plot_boundary(clf, X, y, plot_title):
xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
np.linspace(-3, 3, 50))
clf.fit(X, y)
# plot the decision function for each datapoint on the grid
Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
Z = Z.reshape(xx.shape)

image = plt.imshow(Z, interpolation='nearest',
extent=(xx.min(), xx.max(), yy.min(), yy.max()),
aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.xlabel(r'$<!-- math>$inline$x_1$inline$</math -->$')
plt.ylabel(r'$<!-- math>$inline$x_2$inline$</math -->$')
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.title(plot_title, fontsize=12);


# In[ ]:


plot_boundary(LogisticRegression(), X, y,
"Logistic Regression, XOR problem")


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# In[ ]:


logit_pipe = Pipeline([('poly', PolynomialFeatures(degree=2)), 
('logit', LogisticRegression())])


# In[ ]:


plot_boundary(logit_pipe, X, y,
"Logistic Regression + quadratic features. XOR problem")

