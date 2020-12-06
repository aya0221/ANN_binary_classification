# Simple ANN for predicting if customer will close the bank account
# It's a binary classification problem

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder # for label encoding
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder # for one-hot encoding
from sklearn.metrics import confusion_matrix, accuracy_score

#------------------------------------------------------------------
#Step 1 - Data Preprocessing

##loading the dataset and separating features (X) and label (y)

dataset = pd.read_csv('../data/Churn_Modelling.csv')

print('Done reading the data')

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#print(dataset)
#print(X)
#print(y)

##Encoding categorical data
#Label Encoding the "Gender" column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#print(X)

#One Hot Encoding the "Geography" column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#print(X)


##Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


##Feature Scaling (using inbuilt standard technique)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('Done preprocessing the data')

#------------------------------------------------------------------
#Step 2 - Building the ANN


## Initializing the ANN
ann = tf.keras.models.Sequential()

## Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

## Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

## Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#------------------------------------------------------------------
# Step 3 - Training the ANN

## Compiling the ANN (using binary_crossentropy loss function as the goal is binary classification)
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#------------------------------------------------------------------
# Step 4 - Evaluating the model

## Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

