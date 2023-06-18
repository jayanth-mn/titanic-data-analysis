#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

titanic_data = pd.read_csv("titanic.csv")
titanic_data.head(10)


# In[3]:


print("No. of passengers in original dataset:"+str(len(titanic_data.index)))


# In[5]:


#analysing data


# In[7]:


sns.countplot(x="Survived", data=titanic_data)


# In[8]:


sns.countplot(x="Survived", hue="Sex", data=titanic_data)


# In[9]:


sns.countplot(x="Survived", hue="Pclass", data=titanic_data)


# In[10]:


titanic_data["Age"].plot.hist()


# In[13]:


titanic_data["Fare"].plot.hist(bin=20, figsize=(10,5))


# In[15]:


titanic_data.info()


# In[16]:


sns.countplot(x="SibSp", data=titanic_data)


# # Data Wrangling

# In[18]:


titanic_data.isnull()


# In[19]:


titanic_data.isnull().sum()


# In[22]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis" ) 


# In[23]:


sns.boxplot(x="Pclass",y="Age",data=titanic_data)


# In[24]:


titanic_data.head(5)


# In[25]:


titanic_data.drop("Cabin", axis=1, inplace=True)


# In[26]:


titanic_data.head(5)


# In[27]:


titanic_data.dropna(inplace=True)


# In[28]:


titanic_data.head(5)


# In[30]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False)


# In[31]:


titanic_data.isnull().sum()


# In[32]:


titanic_data.head(5)


# In[39]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head(5)


# In[41]:


embark=pd.get_dummies(titanic_data["Embarked"], drop_first=True)
embark.head(5)


# In[42]:


Pcl=pd.get_dummies(titanic_data["Pclass"], drop_first=True)
Pcl.head(5)


# In[43]:


titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)


# In[44]:


titanic_data.head(5)


# In[48]:


titanic_data.drop(["Sex","Embarked","PassengerId","Name","Ticket"],axis=1,inplace=True)


# In[49]:


titanic_data.head()


# In[50]:


titanic_data.drop(["Pclass"],axis=1,inplace=True)


# In[51]:


titanic_data.head()


# # Train data

# In[52]:


X= titanic_data.drop("Survived", axis=1)
y=titanic_data["Survived"]


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)


# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


logmodel=LogisticRegression()


# In[73]:


logmodel.fit(X_train,y_train)


# In[74]:


predictions = logmodel.predict(X_test)


# In[76]:


from sklearn.metrics import classification_report


# In[77]:


classification_report(y_test, predictions)


# In[78]:


from sklearn.metrics import confusion_matrix


# In[79]:


confusion_matrix(y_test, predictions)


# In[81]:


from sklearn.metrics import accuracy_score


# In[83]:


accuracy_score(y_test, predictions)*100


# In[ ]:




