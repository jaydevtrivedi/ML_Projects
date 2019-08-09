#  Convolution Neural Network
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Step 1: Imports
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Step 2: Get all data using datagenerators
# Dir structure - training_set-> cats, dogs && test_set->cats, dogs
CLASS_0 = "dogs"
CLASS_1 = "cats"
DATASET_TRAINING = "ml_projects_datasets\\cats_v_dogs\\training_set"
DATASET_TEST = "ml_projects_datasets\\cats_v_dogs\\test_set"
DATASET_TRAIN_CLASS_0 = os.path.join(DATASET_TRAINING, CLASS_0)
DATASET_TRAIN_CLASS_1 = os.path.join(DATASET_TRAINING, CLASS_1)
DATASET_TEST_CLASS_0 = os.path.join(DATASET_TEST, CLASS_0)
DATASET_TEST_CLASS_1 = os.path.join(DATASET_TEST, CLASS_1)

NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
LOSS = "binary_crossentropy"
METRICS = "accuracy"

STEPS_PER_EPOCH = round(
    (len(os.listdir(DATASET_TRAIN_CLASS_0)) + len(os.listdir(DATASET_TRAIN_CLASS_1))) / BATCH_SIZE) + 1

VALIDATION_STEPS = round(
    (len(os.listdir(DATASET_TEST_CLASS_0)) + len(os.listdir(DATASET_TEST_CLASS_1))) / BATCH_SIZE) + 1

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(DATASET_TRAINING,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    seed=0)

validation_generator = test_datagen.flow_from_directory(DATASET_TEST,
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary',
                                                        seed=0)

# Step 3 : Build a model and compile
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
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss=LOSS,
              metrics=[METRICS])

history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS, verbose=1,
                              validation_data=validation_generator)

print(np.mean(history.history['accuracy']))
print(np.mean(history.history['val_accuracy']))
