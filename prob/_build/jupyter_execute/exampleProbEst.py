#!/usr/bin/env python
# coding: utf-8

# # Estimation of Probabilities from Datasets
# In this notebook a small dataset of employess is given. Each employee is described by:
# * sex: *m* for male and *f* for female
# * number of years in the company: integer
# * income: *h* (high), *m* (medium) or *l* (low)
# * division: *s* (sales), *d* (design), *b* (backoffice) and *m* (marketing)
# 
# The dataset is defined below and represented as a pandas dataframe.

# In[1]:


import pandas as pd

dataDict={"sex":["m","m","f","m","f","f","f","m","m","m"],
          "years":[10,2,4,4,5,1,7,2,4,1],
          "income":["h","m","m","l","m","l","m","m","h","m"],
          "division":["s","d","d","b","b","m","s","d","d","d"]
         }
data=pd.DataFrame(dataDict)
data


# The following probabilities shall be estimated from the given dataset:
# 1. propability for *male* and *high* income -> $P(male,high)$
# 2. probability for *male*, *low* income and *backoffice* -> $P(male,low,backoffice)$
# 3. probability that a *male* has *high* income -> $P(high|male)$
# 4. probability that a *female* has *high* income -> $P(high|female)$
# 5. probability that an employee with *high* income is *female* -> $P(female|high)$
# 6. probability that a *male* with *medium* income works in division *design* -> $P(design|male,medium)$
# 7. probability that an employee in division *design* is a *male* with *high* income -> $P(male,medium|design)$
# 8. probability that a *male* which is at least 4 years in the company has *medium* income -> $P(medium|male,\geq4)$

# For calculating joint probabilities and conditional probabilities the [pandas method crosstab()](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.crosstab.html) can be applied. **This method creates a table in which the frequencies of all value-combinations of two or more random variables can be determined.** Moreover, by applying the argument *normalize* of the `crosstab()`-method it is possible to calculate instead of the frequencies the joint probabilities or conditional probabilities of all value-combinations. This is demonstrated below: 

# First we calculate the frequencies of all value-combinations of the variables *sex* and *income*: 

# In[2]:


pd.crosstab(data["sex"],data["income"])


# Next we, set the argument `normalize="all"` in the same method call. The result is the complete joint probability distribution of these two variables.

# In[3]:


pd.crosstab(data["sex"],data["income"],normalize="all")


# From the table calculated above, we can derive the answer for question 1:
# 
# $$
# P(male,high)=0.2
# $$

# Next, we set the argument `normalize="index"`. The calculated values are the conditional probabilities $P(income|sex)$:

# In[4]:


pd.crosstab(data["sex"],data["income"],normalize="index")


# The table calculated above contains the solutions for question 2 and 3:
# 
# $$
# P(high|male)=0.333
# $$
# 
# and
# 
# $$
# P(high|female)=0
# $$
# 

# In order to calculate the conditional probabilities of type $P(sex|income)$ we can apply the same `crosstab()`, but now with `normalize="columns"`. 

# In[5]:


pd.crosstab(data["sex"],data["income"],normalize="columns")


# This table contains the answer to question 5:
# 
# $$
# P(female|high)=0
# $$

# The `crosstab()`-method can also be applied for more than two variables, as demonstrated below:

# In[6]:


pd.crosstab([data["sex"],data["income"]],data["division"],normalize="all")


# From the table calculated above, we can derive the answer for question 2:
# 
# $$
# P(male,low,backoffice)=0.1
# $$

# In[7]:


pd.crosstab([data["sex"],data["income"]],data["division"],normalize="index")


# The table calculated above, contains the answer of question 6:
# 
# $$
# P(design|male,medium)=1.0
# $$

# In order to calculate the answer for question 7, we set `normalize="columns"`:

# In[8]:


pd.crosstab([data["sex"],data["income"]],data["division"],normalize="columns")


# From this table we derive the answer for question 7:
# 
# $$
# P(male,medium|design)=0.6
# $$

# For calculating the answer of question 8, we apply the `crosstab()`-method as described below and add the two values in column `m` which belong to rows that belong to *male* and at least 4 years:
# 
# $$
# P(medium|male,\geq 4)= P(medium|male,4) + P(medium|male,10) =0 + 0 = 0 
# $$

# In[9]:


pd.crosstab([data["sex"],data["years"]],data["income"],normalize="index")


# In[ ]:




