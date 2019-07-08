#  Part I data Management
import numpy as np
import pandas as pd

dataset_csv = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\woolies_stock_predict\\stock_prices.csv"

#  Importing the Dataset
dataset_train = pd.read_csv(dataset_csv)
training_set = dataset_train.iloc[:, 1:2].values
training_set = np.flipud(training_set)

#  Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 50 timesteps and 5 output
X_train = []
y_train = []
i = len(training_set_scaled)-1
for i in range(50, 4995):
    X_train.append(training_set_scaled[i-50:i, 0])
    y_train.append(training_set_scaled[i:i+5, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#  Reshaping for the purpose of RNN as it needs to be 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]), 1)

#  Define the r2 score
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    score = r2_score(y_true,y_predict)

    # Return the score
    return score

#  Part 2 - Building the RNN

#  Importing Keras Library and Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

#  Adding regressor to the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=400, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#  Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=400, return_sequences=True))
regressor.add(Dropout(0.2))

#  Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=400, return_sequences=True))
regressor.add(Dropout(0.2))

#  Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=400, return_sequences=True))
regressor.add(Dropout(0.2))

#  Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=400))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=5))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,epochs=200,batch_size=32)

#  Part 3 - Making predictions and visualising the results
X_test = []
y_test = []
X_test.append(training_set_scaled[4995:5045,0])
y_test.append(training_set_scaled[5045:5050, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]), 1)

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
y_pred = np.transpose(y_pred)
y_test = sc.inverse_transform(y_test)
y_test = np.transpose(y_test)

score = performance_metric(y_test, y_pred)
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))
#
# # Visualising the results
# import matplotlib.pyplot as plt
# plt.plot(y_test, color = 'red', label = 'Real Woolies Stock Price')
# plt.plot(y_pred, color = 'blue', label = 'Predicted Woolies Stock Price')
# plt.title('Woolies Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Woolies Stock price')
# plt.legend()
# plt.show()