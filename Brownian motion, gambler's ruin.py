#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


n = 10000
T = 1.
times = np.linspace(0.,T,n)
print(times)


# In[4]:


dt = times[1]-times[0]
# W(t) ~ Normal with mean 0 and variance 1.
dW =np.sqrt(dt) * np.random.normal(size=(n-1,)) 
W0 = np.zeros(shape=(1,))   
W = np.concatenate((W0, np.cumsum(dW,)))
plt.plot(times, W)
plt.show()


# In[5]:


n = 10000
d = 1000
T = 1.
times = np.linspace(0.,T,n)
print(times)


# In[6]:


dt = times[1]-times[0]
# W(t2) - W(t1) ~ Normal with mean 0 and variance t2-t1.
dW =np.sqrt(dt) * np.random.normal(size=(n-1, d)) 
W0 = np.zeros(shape=(1, d))   
W = np.concatenate((W0, np.cumsum(dW, axis = 0)), axis=0)
plt.plot(times, W)
plt.show()


# In[7]:


# Importing the required libraries

import numpy as np

from numpy.linalg import matrix_power

from numpy import linalg  as la

from scipy import linalg  as scla


# In[8]:


#Let's say we have 15 bucks
#each coin toss, we bet $5 on heads
# We will finish the game at 0 or 40

p = 0.5
q = 1 - p

t = np.array([  [1,0,0,0,0,0,0,0,0], 
                [q,0,p,0,0,0,0,0,0], 
                [0,q,0,p,0,0,0,0,0], 
                [0,0,q,0,p,0,0,0,0], 
                [0,0,0,q,0,p,0,0,0], 
                [0,0,0,0,q,0,p,0,0], 
                [0,0,0,0,0,q,0,p,0], 
                [0,0,0,0,0,0,q,0,p], 
                [0,0,0,0,0,0,0,0,1], 
                ]) 

start = np.array([[0,0,0,1,0,0,0,0,0]]) 


# In[9]:


print(np.round(t,25))

n= 1000
# for i in range(2) :
p_1000 = np.round(matrix_power(t, n), 2)  


# In[10]:


print (n, p_1000) 
p_n =np.matmul(start, p_1000)


# In[11]:


print(n, p_n)


# In[12]:


# how many tosses until we go broke ?
import random
random.random()


# In[13]:


def integer_walk():
    money, count = 15,0
    while money:
        toss = random.random()
        if toss < 0.50:
            money -= 5
        if toss > 0.50:
            money += 5
        count += 1
    return count


# In[14]:


# Simulation 1000
sims = [integer_walk() for x in range(1000)]
print(len(sims))


# In[15]:


# Number of times coin rolled out before you ran out of money
sims


# In[16]:


np.median(sims)


# In[17]:


np.mean(sims)


# In[18]:


np.percentile(sims, np.arange(0,100,25))


# In[ ]:




