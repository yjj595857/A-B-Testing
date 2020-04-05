#!/usr/bin/env python
# coding: utf-8

# Marketing Analytics
# <br>Assignment 1 - A/B Testing
# <br>Yujung Huang

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats as st


# In[2]:


data = pd.read_csv("AB_test_data.csv")
data


# In[3]:


data.id.nunique()


# In[4]:


data.isnull().sum()


# In[5]:


data.head()


# ## Is B better than A?

# In[6]:


# number of samples
n_A = data.query('Variant=="A"').shape[0]
n_B = data.query('Variant=="B"').shape[0]
print(n_A)
print(n_B)


# In[7]:


# number of conversion
convert_A = data.query('Variant=="A" & purchase_TF=="1"').shape[0]
convert_B = data.query('Variant=="B" & purchase_TF=="1"').shape[0]
print(convert_A,convert_B)


# In[8]:


# convert rate
p_A = convert_A / n_A
p_B = convert_B / n_B

print(p_A, p_B)


# In[9]:


from scipy.stats import norm
z_alpha = st.norm.ppf(1-0.05/2)
z_alpha


# In[10]:


z = (p_B-p_A) / np.sqrt(p_A*(1-p_A)/n_B)
z


# In[11]:


z > z_alpha


# Because z > z_alpha, we can reject H0: Conversion rate of alternative A >= Conversion rate of alternative B and accept that alternative B improved the converision rate.

# ## Solving for optimal sample size

# In[12]:


from scipy.stats import norm
z_beta = st.norm.ppf(1-0.2)
print(z_beta)


# In[13]:


p_hat = (p_A+p_B)/2
p_hat


# In[14]:


n = (z_alpha*np.sqrt(2*p_hat*(1-p_hat))+z_beta*np.sqrt(p_A*(1-p_A)+p_B*(1-p_B)))**2 * 1/(p_B-p_A)**2 
print("Optimal sample size is %.2f" % n)


# ## Conduct the test 10 times

# In[15]:


bcr = p_A  # baseline conversion rate
d_hat = p_B-p_A # difference between the groups
print(bcr, d_hat)


# In[16]:


def create_df(N_concat):
    
    alist=[]
    
    convert_N_A = N_concat.query('Variant=="A" & purchase_TF=="1"').shape[0]
    convert_N_B = N_concat.query('Variant=="B" & purchase_TF=="1"').shape[0]
    p_N_A = convert_N_A / 50000
    p_N_B = convert_N_B / 1155
    
    z_alpha = st.norm.ppf(1-0.05/2)
    z = (p_N_B-p_N_A) / np.sqrt(p_N_A*(1-p_N_A)/1155)
    
    reject_H0 = z > z_alpha 
    
    alist.append(p_N_B)
    alist.append(z)
    alist.append(reject_H0)
    
    return alist
    


# In[17]:


adict={}
i=0

while i < 10:
    N_A = data[data["Variant"]=="A"]
    N_B = data[data["Variant"]=="B"].sample(n=1155)
    N_concat = pd.concat([N_A,N_B])
    adict[i]=create_df(N_concat)
    i += 1


# In[18]:


df = pd.DataFrame(adict)
df = df.T
df.rename(columns={0:'p_sample', 1:'z_score', 2:'reject_H0'},inplace=True)
df


# ## Conduct a sequential test for the 10 samples

# In[19]:


# upper bound
A = np.log(1/0.05)

# lower bound
B = np.log(0.2)

print(round(A,2),round(B,2))


# In[21]:


import random
import math

samples=[]

i=0
while i < 10:
    samples.append(random.sample(data[data['Variant']=="B"]['id'].tolist(),math.ceil(n)))
    i += 1


# In[22]:


stop_observation = []
stop_criteria = []

for sample_n in samples:
    results = data[data['id'].isin(sample_n)]['purchase_TF'].values
    
    log_gamma = 0
    count = 0
    while (log_gamma > B) & (log_gamma < A):
        if results[count] == True:
            log_gamma = log_gamma + math.log(p_B / p_A)
        else:
            log_gamma = log_gamma + math.log( (1-p_B) / (1-p_A))

        count += 1

    stop_observation.append(count)
    
    if log_gamma < B:
        stop_criteria.append('Lower bound')
    else:
        stop_criteria.append('Upper bound')


# In[24]:


print(stop_observation)
print(stop_criteria)
print("Average iterations: %.2f" % (sum(stop_observation) / len(stop_observation)))

