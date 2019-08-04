# Params
test_size = 0.1
batch_size = 32
dataset_shuffle = True
dataset_file = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\churn_modelling\\Churn_Modelling.csv"

# Step 1 : Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Step 2 : Get the data
dataframe = pd.read_csv(dataset_file)
# RowNumber and CustomerId is non required data for learning
dataframe = dataframe.drop(['RowNumber', 'CustomerId'], axis=1)
print(dataframe.columns)
print(dataframe['HasCrCard'].unique())

# Step 3 : Split data into train, validation and test
train, test = train_test_split(dataframe, test_size=test_size, random_state=0)
train, val = train_test_split(train, test_size=test_size, random_state=0)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# Step 4 : Create an input pipeline using tf.data
def df_to_dataset(dataset, shuffle=dataset_shuffle, batch_size=batch_size):
    dataset = dataset.copy()
    labels = dataset.pop('Exited')
    dataset = sc.fit_transform(dataset)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataset))
    ds = ds.batch(batch_size)
    return ds


# Step 5 : understand input pipeline
batch_size = 1  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of credit scores:', feature_batch['CreditScore'])
    print('A batch of targets:', label_batch)

# Demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]


def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


estSalary = feature_column.numeric_column('EstimatedSalary')
demo(estSalary)

# Step 6 : Choose which columns to use
feature_columns = []

# numeric cols
for header in ['CreditScore', 'Balance', 'NumOfProducts', 'EstimatedSalary']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age = feature_column.numeric_column('Age')
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

tenure = feature_column.numeric_column('Tenure')
tenure_buckets = feature_column.bucketized_column(tenure, boundaries=[0, 3, 5, 7, 9])
feature_columns.append(tenure_buckets)

# indicator cols
geography = feature_column.categorical_column_with_vocabulary_list(
    'Geography', ['France', 'Spain', 'Germany'])
geography_one_hot = feature_column.indicator_column(geography)
gender = feature_column.categorical_column_with_vocabulary_list(
    'Gender', ['Female', 'Male'])
gender_one_hot = feature_column.indicator_column(gender)

feature_columns.append(geography_one_hot)
feature_columns.append(gender_one_hot)

for header in ['HasCrCard', 'IsActiveMember']:
    col = feature_column.categorical_column_with_identity(key=header, num_buckets=2)
    col_one_hot = feature_column.indicator_column(col)
    feature_columns.append(col_one_hot)

# embedding cols
# hashed feature cols
# crossed cols

# Step 7 : Create a feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Step 8 : Create, compile, and train the model
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
