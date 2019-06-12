#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


# In[49]:


x = np.linspace(1, 30)


# In[50]:


plt.plot(x, f(x));


# In[51]:


from scipy.optimize import minimize


# In[52]:


x0 = [2]


# In[53]:


minimize(f, x0, method='BFGS')


# In[54]:


x1 = [30]


# In[55]:


minimize(f, x1, method='BFGS')


# In[56]:


from scipy.optimize import differential_evolution


# In[57]:


differential_evolution(f, [(1, 30)])


# In[79]:


def h(x):
    return np.int_(f(x))


# In[80]:


plt.plot(x, h(x));


# In[81]:


minimize(h, x1, method='BFGS')


# In[82]:


differential_evolution(h, [(1, 30)])

