#!/usr/bin/env python
# coding: utf-8

# # Kullback-Leibler Divergence
# 
# The KL-divergence is an asymmetric statistical distance measure of how much one probability distribution P differs from a reference distribution Q. For continuous random variables the KL-divergence is calculated as follows
# 
# $$
# D_{KL}(P||Q)= \int_{-\infty}^{\infty}p(x) \cdot \log\left( \frac{p(x)}{q(x)} \right) dx,
# $$
# 
# where $p(x)$ and $q(x)$ are the probability density functions of the two distributions $P$ and $Q$.
# 
# For discrete random variables the KL-divergence is calculated as follows
# 
# $$
# D_{KL}(P||Q)= \sum\limits_{i}P(i) \cdot \log\left( \frac{P(i)}{Q(i)} \right).
# $$
# In the discrete case the KL-divergence can only be calculated if $Q(i) > 0$ for all $i$ with $P(i)>0$.
# 
# Below the KL-divergence of two Gaussian distributions is calculated and visualized.

# In[1]:


import numpy as np
import plotly.express as px


# Define the x-axis range:

# In[2]:


dx=0.05
x=np.arange(-10,10,dx)


# Define the first univariate Gaussian distribution:

# In[3]:


mu1=-1
sigma1=3
dist1= 1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp( - (x - mu1)**2 / (2 * sigma1**2))


# In[4]:


ax=px.line(x=x,y=dist1,title="Distribution 1: p(x)")
ax.show()


# Define the second univariate Gaussian distribution:

# In[5]:


mu2=1
sigma2=4
dist2= 1/(sigma2 * np.sqrt(2 * np.pi)) * np.exp( - (x - mu2)**2 / (2 * sigma2**2))


# In[6]:


ax=px.line(x=x,y=dist2,title="Distribution 2: q(x)")
ax['data'][0]['line']['color']='red'
ax.show()


# Kullback-Leibler- Divergence of $p_1(x)$ with respect to $p_2(x)$:

# In[7]:


klsing=dist1*np.log2(dist1/dist2)
kl=np.sum(np.abs(klsing*dx))
print("Kullback-Leibler Divergence: ",kl)


# Visualization of the Kullback-Leibler Divergence. The KL-divergence is the area under the green curve:

# In[8]:


ax=px.line(x=x,y=[dist1,dist2],labels=["p(x)","q(x)"])
ax.add_scatter(x=x,y=klsing,fill="tozerox",opacity=0.2)
ax['data'][0]['name']="p(x)"
ax['data'][1]['name']="q(x)"
ax['data'][2]['name']="p(x)*log(p(x)/q(x))"
ax.show()


# In[ ]:




