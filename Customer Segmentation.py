#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import numpy as np


# In[200]:


df = pd.read_csv(r'C:\Users\user\Desktop\train.csv')


# In[201]:


df


# In[202]:


df.info()


# In[203]:


df.duplicated().sum()


# In[204]:


df.isnull().sum()


# In[205]:


from sklearn.impute import KNNImputer


# In[206]:


impute = KNNImputer()


# In[207]:


for i in df.select_dtypes(include = 'number').columns:
    df[i] = impute.fit_transform(df[[i]])


# In[208]:


df


# In[209]:


df.info()


# In[210]:


for i in df.select_dtypes(include = object).columns:
    df[i] = df[i].fillna(df[i].mode()[0])


# In[211]:


df.info()


# In[212]:


from sklearn.preprocessing import LabelEncoder


# In[213]:


le = LabelEncoder()


# In[214]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = le.fit_transform(df[i])


# In[215]:


df.info()


# In[216]:


df


# In[217]:


df = df.select_dtypes(include = 'number').astype('int64')


# In[218]:


df.info()


# In[219]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler


# In[220]:


mm = MinMaxScaler()


# In[221]:


scaled = mm.fit_transform(df)


# In[222]:


scaled


# In[223]:


df = pd.DataFrame(scaled, columns=df.columns, index=df.index)


# In[224]:


df


# In[225]:


df.info()


# In[226]:


from sklearn.model_selection import train_test_split


# In[227]:


x = df.drop(columns = 'Segmentation')
y = df['Segmentation']


# In[228]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[229]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[230]:


from sklearn.ensemble import GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(x_train, y_train)
y_pred = gbr_model.predict(x_test)



# In[231]:


y_pred


# In[232]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# In[ ]:




