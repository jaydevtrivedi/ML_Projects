#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:49:56 2019

@author: jaydevtrivedi
"""
#Params
test_size = 0.1
batch_size = 32
dataset_shuffle = True
dataset_file = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\churn_modelling\\Churn_Modelling.csv"

# Step 1 : Import the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# Step 2 : Get the data
dataset = pd.read_csv(dataset_file)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Step 3:  Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Step 4 : Split data into train, validation and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=0)
print(len(X_train), 'train examples')
print(len(X_val), 'validation examples')
print(len(X_test), 'test examples')

# Step 5 : Scale data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train[1].shape)

# Step 6 : Create, compile, and train the model
def build_classifier(optimizer):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(128, activation='relu', input_shape=X_train[1].shape))
    classifier.add(tf.keras.layers.Dropout(0.2))
    classifier.add(tf.keras.layers.Dense(128, activation='relu'))
    classifier.add(tf.keras.layers.Dropout(0.1))
    classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32, 64],
              'epochs': [100, 250, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)