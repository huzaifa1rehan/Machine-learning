#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r'C:\Users\user\Desktop\stock_price_dataset.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


for i in df.select_dtypes(include = 'number').columns:
    df[i] = df[i].fillna(df[i].mean())
    


# In[7]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = df[i].fillna(df[i].mode()[0])


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


le = LabelEncoder()


# In[12]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = le.fit_transform(df[i])


# In[13]:


df.info()


# In[14]:


df = df.select_dtypes(include = 'number').astype('int64')


# In[15]:


df


# In[16]:


df.info()


# In[17]:


df


# In[19]:


df = df.drop(columns = ['Date','Comments'])


# In[20]:


df


# In[21]:


df.duplicated().sum()


# In[22]:


df.drop_duplicates(inplace = True)


# In[23]:


df


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x = df.drop(columns = 'Sentiment_Score')
y = df['Sentiment_Score']


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[27]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[28]:


from sklearn.svm import SVC


# In[29]:


model = SVC()


# In[31]:


model


# In[37]:


model.fit(x_train,y_train)


# In[38]:


y_pred = model.predict(x_test)


# In[39]:


y_pred


# In[41]:


y_pred[:50]


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


accuracy = accuracy_score(y_pred,y_test)
print("Accuracy:",accuracy)


# In[44]:


print(f"Model Accuracy: {accuracy:.2f}%")


# In[ ]:




