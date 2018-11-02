#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:05:54 2018

@author: fatinhalim
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import pickle
print('Libraries Imported')


dataset = pd.read_csv('TEST_DATA_2.csv', header = None)
dataset.columns = ['Name','LEARN','ML_CATEGORIES','ML_CATEGORIES_2','PROGRAMMING','PYTHON','DATA_STRUCTURE','DS_Q1','DS_Q2','STATISTICS','STAT_Q1','RATE']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())

dataset['DS_Q2_ans'] = np.where(dataset['DS_Q2']== 'linear search', 'yes', 'no')
dataset['STAT_1_ans'] = np.where(dataset['STAT_Q1']== '{(1, a), (2, a), (1, b), (2, b)}', 'yes', 'no')

#Creating the dependent variable class
factor = pd.factorize(dataset['RATE'])
dataset.RATE = factor[0]
definitions = factor[1]
print(dataset.RATE.head())
print(definitions)

#Splitting the data into independent and dependent variables
dummy_df = pd.get_dummies(dataset[['LEARN','PROGRAMMING','DS_Q2_ans','PYTHON','DATA_STRUCTURE', 'STATISTICS', 'STAT_1_ans']])
X = dummy_df.iloc[:, [1,4,5,6,9,12,13]].values #independent variable or #[0,2,4,6]
#X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,11].values
print('The independent features set: ')
print(X[:5,:])
print('The dependent variable: ')
print(y[:5])

# Encoding categorical data/Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
dummy_df.values[:, 0] = labelencoder_X.fit_transform(dummy_df.values[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
dummy_df = onehotencoder.fit_transform(X).toarray()

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Reverse factorize 
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Level'], colnames=['Predicted Student Level']))

print(list(zip(dataset.columns[0:13], classifier.feature_importances_)))
joblib.dump(classifier, 'randomforest_new_data.pkl', compress = 1) 

