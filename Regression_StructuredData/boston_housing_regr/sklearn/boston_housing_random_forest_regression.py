# Step 1 : Let's import the data
dataset_csv = "C:\\Users\\Jaydev\\Documents\\Datasets\\tf_in_practice_datasets\\boston_housing\\housing.csv"

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# Fitting SVR to the dataset
def random_forest_regressor():
    model = RandomForestRegressor(random_state=0)
    # parameters = {'n_estimators': [300, 500, 700], 'criterion': ['mse', 'mae', 'friedman_mse'],
    #               'max_depth': [1, 3, 5, 7, 9], 'min_samples_split': [4, 8, 16],
    #               'random_state': [0, 3], 'min_impurity_split': [0.1, 0.3],
    #               'bootstrap': [True, False], 'warm_start': [True, False]}
    #                 'min_impurity_decrease ': [0.3, 0.9],'max_leaf_nodes ': [3, 5, 7, 9, 11],
    #                 'max_features ': ['auto', 'sqrt', 'log2'], 'min_samples_leaf ': [],
    #                 'min_weight_fraction_leaf ': [3, 5, 7, 9],
    parameters = {}
    grid_search = GridSearchCV(model, parameters, cv=5)
    return grid_search


# Step 5 : Fit the data
regressor = random_forest_regressor()
regressor.fit(X_train, y_train)

# Step 6 :  R2 Score
from sklearn.metrics import r2_score

print("R2 score is {}".format(r2_score(y_test, regressor.predict(X_test))))

# Step 6 : Predict some data
print(standard_scalar_target.inverse_transform(regressor.predict(X_predict)))
print("Actual : ")
print(standard_scalar_target.inverse_transform(y_predict))
