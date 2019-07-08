# Step 1 : Let's import the data
dataset_csv = "C:\\Users\\Jaydev\\Documents\\Datasets\\tf_in_practice_datasets\\boston_housing\\housing.csv"
INPUT_DIMENSIONS = 3

import pandas as pd

dataset = pd.read_csv(dataset_csv)

# Step 2 : Explore the data
dataset.head()
dataset.describe()
dataset.corr()

# Step 3 : Shuffle, Scale and Spilt the data
dependent_columns = ['RM', 'LSTAT', 'PTRATIO']
target_columns = ['MEDV']
dependent_variables = dataset[dependent_columns]
target_variable = dataset[target_columns]

from sklearn.preprocessing import StandardScaler
standard_scalar_dependent = StandardScaler()
standard_scalar_target = StandardScaler()
dependent_variables = standard_scalar_dependent.fit_transform(dependent_variables)
target_variable = standard_scalar_target.fit_transform(target_variable)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dependent_variables, target_variable, train_size=0.9,
                                                    random_state=0)
X_train, X_predict, y_train, y_predict = train_test_split(X_train, y_train, train_size=0.99,
                                                    random_state=0)

# Step 4 : Build a model
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_regressor(optimizer):
    model = models.Sequential()
    model.add(layers.Dense(units=128, activation='relu', input_dim=INPUT_DIMENSIONS))
    model.add(layers.Dense(units=1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

#  Implementing GridSearchCV
regressor = KerasRegressor(build_fn=build_regressor)
parameters = {'batch_size' : [10,25,32,64],
              'epochs' : [100,150,300],
              'optimizer' : ['adam','rmsprop','sgd']}
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10)

# Step 5 : Fit the data
history = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Step 6 :  R2 Score
from sklearn.metrics import r2_score
print("R2 score is {}".format(r2_score(y_test,regressor.predict(X_test))))

# Step 6 : Predict some data
print(standard_scalar_target.inverse_transform(regressor.predict(X_predict)))
print("Actual : ")
print(standard_scalar_target.inverse_transform(y_predict))
