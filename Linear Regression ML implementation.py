#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[15]:


from sklearn.datasets import fetch_california_housing


# In[18]:


df=fetch_california_housing()


# In[22]:


df.keys()


# In[23]:


print(df.DESCR)


# In[24]:


print(df.feature_names)


# In[25]:


dataset=pd.DataFrame(df.data)


# In[30]:


dataset.head()


# In[29]:


dataset['MedInc']=  df.target


# In[31]:


dataset.info()


# In[32]:


dataset.describe()


# In[33]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[34]:


x.head()


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[37]:


x_test


# In[39]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[42]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)


# In[46]:


reg_pred=regression.predict(x_test)


# In[47]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
score=mean_squared_error(y_test,reg_pred)
print(score)


# In[50]:


scaler.transform(df.data[0].reshape(1,-1))


# In[49]:


regression.predict(df.data[0].reshape(1,-1))


# In[52]:


import pickle
pickle.dump(regression,open('regmodel.pkl','wb'))


# In[53]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[54]:


pickled_model.predict(scaler.transform(df.data[0].reshape(1,-1)))

import
pickle.dump(scaler,open('scaling.pkl','wb'))