# Recurrent Neural Network
NUM_OF_EPOCHS = 100
BATCH_SIZE = 32

# Step 1: All Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 2 :  Get the data
dataset_train_file = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\stock_price_google\\google_sp_train.csv"
dataset_test_file = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\stock_price_google\\google_sp_test.csv"

# Importing the training set
dataset_train = pd.read_csv(dataset_train_file)
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X = []
y = []
for i in range(60, 1258):
    X.append(training_set_scaled[i - 60:i, 0])
    y.append(training_set_scaled[i, 0])
X, y = np.array(X), np.array(y)

print(X.shape)
print(y.shape)

# Reshaping
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(X.shape)
print(y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, random_state=0)

print(X_train.shape)
print(y_train.shape)


# Step 3 : Building the model
def single_layer_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def multi_layer_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def conv1D():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, 5, activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def multilayer_gru():
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


list_func = [single_layer_lstm, multi_layer_lstm, conv1D, multilayer_gru]
dict_results = dict()

for f in list_func:
    model = f()
    model.fit(X_train, y_train, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    # Step 4 : Getting the real data and comparing predictions
    dataset_test = pd.read_csv(dataset_test_file)
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Getting the predicted stock price of 2017
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    dict_results[f.__name__] = r2_score(real_stock_price, predicted_stock_price)

print(dict_results)

# Best result till now
# bs = 1, epochs = 100
# {'single_layer_lstm': -0.44437182141843135, 'multi_layer_lstm': 0.13769778661582666, 'conv1D': -6.6136835861118595, 'multilayer_gru': -0.22029392469808506}

# Step 5 :  Visualising the results
# plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()
