#!/usr/bin/env python
# coding: utf-8

# # Simple ANN for predicting if customer will close the bank account

# ### Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# ## Step 1 - Data Preprocessing

# ### loading the dataset and separating features (X) and label (y)

# In[3]:


dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[ ]:


print(dataset)


# In[4]:


print(X)


# In[5]:


print(y)


# ### Encoding categorical data

# Label Encoding the "Gender" column

# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[7]:


print(X)


# One Hot Encoding the "Geography" column

# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[9]:


print(X)


# ### Splitting the dataset into Training set and Test set

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Feature Scaling (using inbuilt standard technique)

# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Step 2 - Building the ANN

# ### Initializing the ANN

# In[12]:


ann = tf.keras.models.Sequential()


# ### Adding the input layer and the first hidden layer

# In[13]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the second hidden layer

# In[14]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the output layer

# In[15]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Step 3 - Training the ANN

# ### Compiling the ANN (using binary_crossentropy loss function as the goal is binary classification)

# In[16]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the ANN on the Training set

# In[17]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# ## Step 4 - Evaluating the model

# ### Predicting the Test set results

# In[19]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Making the Confusion Matrix

# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

