#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv(r'C:\Users\user\Desktop\machine failure.csv')


# In[3]:


df


# In[4]:


df = df.rename(columns={'Air temperature [K]': 'Air temperature K', 'Process temperature [K]': 'Process temperature K'})
df = df.rename(columns={'Rotational speed [rpm]': 'Rotational speed rpm', 'Torque [Nm]': 'Torque Nm'})
df = df.rename(columns={'Tool wear [min]': 'Tool wear min'})


# In[5]:


df


# In[6]:


df.drop('UDI', axis=1, inplace=True)


# In[7]:


df


# In[8]:


df.drop('Product ID', axis=1, inplace=True)
df


# In[9]:


df.duplicated().sum()


# In[10]:


df.info()


# In[11]:


df


# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


le = LabelEncoder()


# In[16]:


for i in df.select_dtypes(include = 'object').columns:
    df[i] = le.fit_transform(df[i])


# In[17]:


df


# In[18]:


df.dtypes


# In[19]:


df.std()


# In[20]:


from scipy import stats
z_scores = stats.zscore(df)
df_cleaned = df[(z_scores <3).all(axis = 1)]
df_cleaned


# In[21]:


df.std()


# In[22]:


df.info()


# In[23]:


df = df.select_dtypes(include = 'number').astype('int64')
df


# In[24]:


df


# In[25]:


df.dtypes


# In[26]:


x = df.drop(columns = 'Machine failure')
y = df['Machine failure']


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[29]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[30]:


import xgboost as xgb


# In[31]:


model = xgb.XGBClassifier(eval_metric='mlogloss')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[32]:


y_pred


# In[33]:


xgb_model = xgb.XGBClassifier(
    max_depth=3,        
    min_child_weight=5,    
    n_estimators=100         
)
xgb_model


# In[34]:


from sklearn.metrics import accuracy_score


# In[35]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[36]:


from sklearn.model_selection import cross_val_score
import xgboost as xgb

model = xgb.XGBClassifier()
scores = cross_val_score(model, x, y, cv=5)

print(f"Cross-validation Scores: {scores}")
print(f"Mean Cross-validation Accuracy: {scores.mean()}")


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(data=x_train)
plt.show()


# In[38]:


df = df.to_csv('C:\\Users\\user\\Desktop\\submission.csv',index = False)
print('Submission saved')


# In[ ]:




