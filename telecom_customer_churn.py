#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np


# In[28]:


df = pd.read_csv(r'C:\Users\user\Desktop\telecom_customer_churn.csv')


# In[29]:


df


# In[30]:


df.info()


# In[31]:


df.duplicated().sum()


# In[32]:


df.isnull().sum()


# In[33]:


from sklearn.impute import KNNImputer


# In[34]:


impute = KNNImputer()


# In[35]:


for i in df.select_dtypes(include = 'number').columns:
    df[i] = impute.fit_transform(df[[i]])


# In[36]:


df


# In[37]:


df.info()


# In[38]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = df[i].fillna(df[i].mode()[0])


# In[39]:


df.info()


# In[40]:


from sklearn.preprocessing import LabelEncoder


# In[41]:


le = LabelEncoder()


# In[42]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = le.fit_transform(df[i])


# In[43]:


df.info()


# In[44]:


df = df.select_dtypes(include = 'number').astype('int64')


# In[45]:


df


# In[46]:


df.info()


# In[48]:


df.std()


# In[50]:


x = df.drop(columns = 'Churn Reason')
y = df['Churn Reason']


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[55]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[56]:


from sklearn.ensemble import RandomForestClassifier


# In[57]:


from sklearn.metrics import classification_report


# In[60]:


model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred


# In[ ]:




