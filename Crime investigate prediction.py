#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np


# In[98]:


df = pd.read_csv(r'C:\Users\user\Desktop\crime_data.csv')


# In[99]:


df


# In[100]:


df.duplicated().sum()


# In[101]:


df = df.drop_duplicates()


# In[102]:


df


# In[103]:


df.isnull().sum()


# In[104]:


df.describe()


# In[105]:


df.info()


# In[106]:


from sklearn.preprocessing import LabelEncoder


# In[107]:


le = LabelEncoder()


# In[108]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = le.fit_transform(df[i])
    
df


# In[109]:


df.std()


# In[110]:


df.info()


# In[111]:


df = df.select_dtypes(include = 'number').astype('int64')
df


# In[112]:


df.info()


# In[133]:


x = df.drop(columns = 'Disposition')
y = df['Disposition']


# In[134]:


from sklearn.model_selection import train_test_split


# In[135]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[136]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[137]:


from sklearn.linear_model import LogisticRegression


# In[138]:


model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[139]:


y_pred


# In[140]:


from sklearn.metrics import accuracy_score, classification_report


# In[141]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:




