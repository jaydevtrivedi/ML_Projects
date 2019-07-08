# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from Database import DataCleaner
from Methods import ClassificationMethods
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dataset_csv = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\aus_rain_pred\\weather.csv"
# Any results you write to the current directory are saved as output.

# Importing the Dataset
dataset = pd.read_csv(dataset_csv)

datacleaner = DataCleaner(dataset)
clean_dataset = datacleaner.deal_with_nans()
X,y = datacleaner.peform_encoding(clean_dataset)

#  Now that the data is clean we shall distribute it into dependent and independent
#  variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Perform Logistic Regression
methods = ClassificationMethods(X_train,X_test,y_train,y_test)
y_pred = methods.logisticRegression()

print(confusion_matrix(y_test,y_pred))
print(r2_score(y_test,y_pred))




