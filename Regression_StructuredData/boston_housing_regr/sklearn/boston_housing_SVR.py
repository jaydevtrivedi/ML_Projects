# Step 1 : Let's import the data
dataset_csv = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\boston_housing\\housing.csv"

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

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Fitting SVR to the dataset
def svr_regressor():
    model = SVR()
    # parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [1, 2, 3, 4],
    #               'gamma': ['auto', 'scale'], 'coef0': [0.0, 0.1, 0.2], 'tol': [0.001, 0.002], 'C': [1.0, 1.1],
    #               'epsilon': [0.1, 0.2, 0.3], 'shrinking': [True, False], 'cache_size': [200, 210, 220],
    #               'verbose': [True, False], 'max_iter': [3,5]}
    parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [1, 2, 3, 4],
                  'gamma': ['auto', 'scale'], 'coef0': [0.0, 0.1, 0.2], 'tol': [0.001, 0.002], 'C': [1.0, 1.1]}
    grid_search = GridSearchCV(model, parameters, cv=5)
    return grid_search


# Step 5 : Fit the data
regressor = svr_regressor()
regressor.fit(X_train, y_train)

# Step 6 :  R2 Score
from sklearn.metrics import r2_score

print("R2 score is {}".format(r2_score(y_test, regressor.predict(X_test))))

# Step 6 : Predict some data
print(standard_scalar_target.inverse_transform(regressor.predict(X_predict)))
print("Actual : ")
print(standard_scalar_target.inverse_transform(y_predict))
