#  Convolution Neural Network
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Step 1: Imports
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Step 2: Declaring all constants
CLASS_0 = "dogs"
CLASS_1 = "cats"
DATASET_TRAINING = "ml_projects_datasets\\cats_v_dogs\\training_set"
DATASET_TEST = "ml_projects_datasets\\cats_v_dogs\\test_set"
DATASET_TRAIN_CLASS_0 = os.path.join(DATASET_TRAINING, CLASS_0)
DATASET_TRAIN_CLASS_1 = os.path.join(DATASET_TRAINING, CLASS_1)
DATASET_TEST_CLASS_0 = os.path.join(DATASET_TEST, CLASS_0)
DATASET_TEST_CLASS_1 = os.path.join(DATASET_TEST, CLASS_1)

TF_CACHE_DIR = "ML_Projects\\cats_v_dogs\\tfcache_dir"
TF_REC_DIR = "ML_Projects\\cats_v_dogs\\tfrec_dir"
file_train = "train"
file_val = "val"
file_test = "test"

NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
LOSS = "binary_crossentropy"
METRICS = "accuracy"

# Step 3 : Get all data(files paths)
# Dir structure - training_set-> cats, dogs && test_set->cats, dogs
def get_all_imgs_paths(dataset_train_class_0, dataset_train_class_1, dataset_test_class_0, dataset_test_class_1):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    zero = tf.cast(0, tf.bfloat16)
    one = tf.cast(1, tf.bfloat16)

    for image in os.listdir(dataset_train_class_0):
        X_train.append(os.path.join(dataset_train_class_0, image))
        y_train.append(zero)
    for image in os.listdir(dataset_train_class_1):
        X_train.append(os.path.join(dataset_train_class_1, image))
        y_train.append(one)

    for image in os.listdir(dataset_test_class_0):
        X_test.append(os.path.join(dataset_test_class_0, image))
        y_test.append(zero)
    for image in os.listdir(dataset_test_class_1):
        X_test.append(os.path.join(dataset_test_class_1, image))
        y_test.append(one)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_val, y_val = get_all_imgs_paths(DATASET_TRAIN_CLASS_0, DATASET_TRAIN_CLASS_1,
                                                    DATASET_TEST_CLASS_0, DATASET_TEST_CLASS_1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.9, random_state=42)
steps_per_epoch = round(len(X_train) / BATCH_SIZE) + 1
val_steps = round(len(X_val) / BATCH_SIZE) + 1
test_steps = round(len(X_test) / BATCH_SIZE) + 1


# Step 4 : Convert all files to datasets tf.data (tensorslices or cache files or tfrec files)
def load_and_preprocess_image(*args):
    multiple_args = len(args) > 1
    if multiple_args:
        path = args[0]
        label = args[1]
    else:
        path = args[0]

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image /= 255
    image = tf.cast(image, tf.bfloat16)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    if multiple_args:
        return image, label
    else:
        return image


# ----------------------------Comment this and use either cache files or tfrec files
# # Using tensor_slices
def get_ds_tensor_slices(X_train, y_train, batch_size, load_preprocess_image):
    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds = ds.map(load_preprocess_image)
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size, seed=0))
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=batch_size)
    return ds


image_ds_train = get_ds_tensor_slices(X_train=X_train, y_train=y_train, batch_size=BATCH_SIZE,
                                      load_preprocess_image=load_and_preprocess_image)

image_ds_validate = get_ds_tensor_slices(X_train=X_val, y_train=y_val, batch_size=BATCH_SIZE,
                                         load_preprocess_image=load_and_preprocess_image)

image_ds_test = get_ds_tensor_slices(X_train=X_test, y_train=y_test, batch_size=BATCH_SIZE,
                                     load_preprocess_image=load_and_preprocess_image)
# ----------------------------Comment this and use either cache files or tfrec files

# ----------------------------Uncomment this and use cache files
# Using cache files
# def get_ds_tfdata_cache(X_train, y_train, batch_size, load_preprocess_image, filename):
#     ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     ds = ds.map(load_preprocess_image)
#     ds = ds.cache(filename=TF_CACHE_DIR + "/" + filename)
#     ds = ds.apply(
#         tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
#     ds = ds.batch(batch_size=batch_size)
#     ds = ds.prefetch(buffer_size=batch_size)
#     return ds
#
#
# image_ds_train = get_ds_tfdata_cache(X_train=X_train, y_train=y_train, batch_size=BATCH_SIZE,
#                                      load_preprocess_image=load_and_preprocess_image, filename=file_train_cache)
#
# image_ds_validate = get_ds_tfdata_cache(X_train=X_val, y_train=y_val, batch_size=BATCH_SIZE,
#                                         load_preprocess_image=load_and_preprocess_image, filename=file_val_cache)
#
# image_ds_test = get_ds_tfdata_cache(X_train=X_test, y_train=y_test, batch_size=BATCH_SIZE,
#                                     load_preprocess_image=load_and_preprocess_image, filename=file_test_cache)
# ----------------------------Uncomment this and use cache files

# ----------------------------Uncomment this and use tfrec files
# Using tfrec files
# def write_ds_tfrec_file(file_paths, load_and_preprocess_image, filename):
#     tfrec_file_path = os.path.join(TF_REC_DIR, filename)
#     tfrec_file_path += ".tfrec"
#
#     ds = tf.data.Dataset.from_tensor_slices(file_paths)
#     ds = ds.map(load_and_preprocess_image)
#     ds = ds.map(tf.io.serialize_tensor)
#     tfrec_file = tf.data.experimental.TFRecordWriter(filename=tfrec_file_path)
#     tfrec_file.write(ds)
#     return tfrec_file
#
#
# image_ds_train = write_ds_tfrec_file(file_paths=X_train,
#                                      load_and_preprocess_image=load_and_preprocess_image, filename=file_train)
#
# image_ds_validate = write_ds_tfrec_file(file_paths=X_val,
#                                         load_and_preprocess_image=load_and_preprocess_image, filename=file_val)
#
# image_ds_test = write_ds_tfrec_file(file_paths=X_test,
#                                     load_and_preprocess_image=load_and_preprocess_image, filename=file_test)
#
#
# def parse(x):
#     result = tf.io.parse_tensor(x, out_type=tf.float32)
#     result = tf.reshape(result, [IMAGE_SIZE, IMAGE_SIZE, 3])
#     return result
#
#
# def get_ds_tfrec_file(label, batch_size, filename):
#     tfrec_file_path = os.path.join(TF_REC_DIR, filename)
#     tfrec_file_path += ".tfrec"
#
#     ds = tf.data.TFRecordDataset(tfrec_file_path)
#     ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     label_ds = tf.data.Dataset.from_tensor_slices(label)
#     ds = tf.data.Dataset.zip((ds, label_ds))
#     ds = ds.apply(
#         tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
#     ds = ds.batch(batch_size=batch_size)
#     ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     return ds
#
#
# image_ds_train = get_ds_tfrec_file(label=y_train,
#                                    batch_size=BATCH_SIZE, filename=file_train)
#
# image_ds_validate = get_ds_tfrec_file(label=y_val,
#                                       batch_size=BATCH_SIZE, filename=file_val)
#
# image_ds_test = get_ds_tfrec_file(label=y_test,
#                                   batch_size=BATCH_SIZE, filename=file_test)
# ----------------------------Uncomment this and use tfrec files

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
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss=LOSS,
              metrics=[METRICS]
              )

history = model.fit(image_ds_train,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=image_ds_validate,
                    validation_steps=val_steps)

print(np.mean(history.history['accuracy']))
print(np.mean(history.history['val_accuracy']))

# Step 6 : Evaluate the model
loss, accuracy = model.evaluate(image_ds_test, steps=test_steps)
print(loss)
print(accuracy)
