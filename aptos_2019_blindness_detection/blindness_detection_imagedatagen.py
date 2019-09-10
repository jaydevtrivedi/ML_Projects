#  Convolution Neural Network
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Step 1: Imports
import tensorflow as tf
import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 2: Declaring all constants
DATASET_BASE_PATH = "C:\\Users\\Jaydev\\Documents\\Datasets\\ml_projects_datasets\\aptos2019_blindness_detection"
DATASET_TRAINING_CSV = os.path.join(DATASET_BASE_PATH, "train.csv")
DATASET_TEST_CSV = os.path.join(DATASET_BASE_PATH, "test.csv")
DATASET_SUBMISSION_CSV = os.path.join(DATASET_BASE_PATH, "sample_submission.csv")
DATASET_TRAIN_IMAGES = os.path.join(DATASET_BASE_PATH, "train_images")
DATASET_TRAIN_IMAGES_DATAGEN = os.path.join(DATASET_BASE_PATH, "image_datagen_train_images")
DATASET_TEST_IMAGES_DATAGEN = os.path.join(DATASET_BASE_PATH, "image_datagen_test_images")
DATASET_TEST_IMAGES = os.path.join(DATASET_BASE_PATH, "test_images")

TF_REC_DIR = "C:\\Users\\Jaydev\\Documents\\GitHub\\repos\\ML_Projects\\aptos_2019_blindness_detection\\tfrec_dir"
file_train = "train"
file_val = "val"
file_test = "test"
file_predict = "predict"

NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
LOSS = "categorical_crossentropy"
METRICS = "categorical_accuracy"

STEPS_PER_EPOCH = round(len(os.listdir(DATASET_TRAIN_IMAGES)) / BATCH_SIZE) + 1
STEPS_PER_EPOCH_PREDICT = round(len(os.listdir(DATASET_TEST_IMAGES)) / BATCH_SIZE) + 1
print(STEPS_PER_EPOCH_PREDICT)

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(DATASET_TRAIN_IMAGES_DATAGEN,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    seed=42)

predict_dir = tf.keras.utils.get_file(
    DATASET_TEST_IMAGES_DATAGEN,
    origin=None,
    untar=False)
predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
predict_generator = predict_datagen.flow_from_directory(predict_dir, class_mode=None,
                                                        shuffle=True, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                        batch_size=BATCH_SIZE)

# Step 5 : Build a model and compile
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',
                           input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss=LOSS,
              metrics=[METRICS]
              )

history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

print(np.mean(history.history['loss']))
print(np.mean(history.history['categorical_accuracy']))

# Step 7 : Predict some to confirm
history_predict = model.predict_generator(predict_generator, steps=STEPS_PER_EPOCH_PREDICT,
                                          verbose=1)
print(sum(history_predict))
# results = np.argmax(history_predict, axis=1)
# print(results[:100])
#
# actual_dataset = pd.read_csv(DATASET_SUBMISSION_CSV)
# print(actual_dataset['diagnosis'])

