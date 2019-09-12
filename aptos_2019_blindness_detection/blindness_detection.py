#  Convolution Neural Network
IMAGE_SIZE = 224
BATCH_SIZE = 30
NUM_CLASSES = 5

# Step 1: Imports
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import models
from tensorflow.keras import layers
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 2: Declaring all constants
DATASET_BASE_PATH = "/media/jaydev/Jay_Dev/Documents/Datasets/ml_projects_datasets/aptos2019_blindness_detection"
TENSORBOARD_DIR = "/media/jaydev/Jay_Dev/Documents/Datasets/ml_projects_datasets/aptos2019_blindness_detection/tensorboard_dir"
MODEL_CHECKPOINT_DIR = "/media/jaydev/Jay_Dev/Documents/Datasets/ml_projects_datasets/aptos2019_blindness_detection/checkpoint_dir"

TF_REC_DIR = "/media/jaydev/Jay_Dev/Documents/GitHub/repos/ML_Projects/aptos_2019_blindness_detection/tfrec_dir"

DATASET_TRAINING_CSV = os.path.join(DATASET_BASE_PATH, "train.csv")
DATASET_TEST_CSV = os.path.join(DATASET_BASE_PATH, "test.csv")
DATASET_SUBMISSION_CSV = os.path.join(DATASET_BASE_PATH, "sample_submission.csv")
DATASET_TRAIN_IMAGES = os.path.join(DATASET_BASE_PATH, "train_images")
DATASET_TEST_IMAGES = os.path.join(DATASET_BASE_PATH, "test_images")

print(DATASET_TRAINING_CSV)

file_train = "train"
file_val = "val"
file_test = "test"
file_predict = "predict"

NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
OPTIMIZER = tf.keras.optimizers.SGD(lr=0.005, momentum=0.9)
LOSS = "sparse_categorical_crossentropy"
METRICS = "sparse_categorical_accuracy"

dataset = pd.read_csv(DATASET_TRAINING_CSV)


# Step 3 : Get all data(files paths)
def get_all_imgs_paths(dataset_csv, dataset_path):
    X = []
    y = []

    dataset = pd.read_csv(dataset_csv)
    image_names = dataset.iloc[:, 0].values
    y = dataset.iloc[:, 1].values
    for name in image_names:
        name = name + ".png"
        X.append(os.path.join(dataset_path, name))

    return X, y


def get_predict_imgs_paths(dataset_csv, dataset_path):
    X = []

    dataset = pd.read_csv(dataset_csv)
    image_names = dataset.iloc[:, 0].values
    for name in image_names:
        name = name + ".png"
        X.append(os.path.join(dataset_path, name))

    return X


X_train, y_train = get_all_imgs_paths(DATASET_TRAINING_CSV, DATASET_TRAIN_IMAGES)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.9, random_state=42)

X_predict = get_predict_imgs_paths(DATASET_TEST_CSV, DATASET_TEST_IMAGES)

steps_per_epoch = round(len(X_train) / BATCH_SIZE) + 1
val_steps = round(len(X_val) / BATCH_SIZE) + 1
test_steps = round(len(X_test) / BATCH_SIZE) + 1
predict_steps = round(len(X_predict) / BATCH_SIZE) + 1


# Step 4 : Convert all files to datasets tf.data (tensorslices or cache files or tfrec files)
def load_and_preprocess_image(*args):
    multiple_args = len(args) > 1
    if multiple_args:
        path = args[0]
        label = args[1]
    else:
        path = args[0]

    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image /= 255
    image = tf.cast(image, tf.bfloat16)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    if multiple_args:
        return image, label
    else:
        return image


# Using tfrec files
def write_ds_tfrec_file(file_paths, load_and_preprocess_image, filename):
    tfrec_file_path = os.path.join(TF_REC_DIR, filename)
    tfrec_file_path += ".tfrec"

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(load_and_preprocess_image)
    ds = ds.map(tf.io.serialize_tensor)
    tfrec_file = tf.data.experimental.TFRecordWriter(filename=tfrec_file_path)
    tfrec_file.write(ds)
    return tfrec_file


image_ds_train = write_ds_tfrec_file(file_paths=X_train,
                                     load_and_preprocess_image=load_and_preprocess_image, filename=file_train)

image_ds_validate = write_ds_tfrec_file(file_paths=X_val,
                                        load_and_preprocess_image=load_and_preprocess_image, filename=file_val)

image_ds_test = write_ds_tfrec_file(file_paths=X_test,
                                    load_and_preprocess_image=load_and_preprocess_image, filename=file_test)

image_ds_predict = write_ds_tfrec_file(file_paths=X_predict,
                                       load_and_preprocess_image=load_and_preprocess_image, filename=file_predict)


def parse(x):
    result = tf.io.parse_tensor(x, out_type=tf.float32)
    result = tf.reshape(result, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return result


def get_ds_tfrec_file(label, batch_size, filename):
    tfrec_file_path = os.path.join(TF_REC_DIR, filename)
    tfrec_file_path += ".tfrec"

    ds = tf.data.TFRecordDataset(tfrec_file_path)
    ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(label)
    ds = tf.data.Dataset.zip((ds, label_ds))
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def get_predict_ds_tfrec_file(batch_size, filename):
    tfrec_file_path = os.path.join(TF_REC_DIR, filename)
    tfrec_file_path += ".tfrec"

    ds = tf.data.TFRecordDataset(tfrec_file_path)
    ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = tf.data.Dataset.zip(ds)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


image_ds_train = get_ds_tfrec_file(label=y_train,
                                   batch_size=BATCH_SIZE, filename=file_train)

image_ds_validate = get_ds_tfrec_file(label=y_val,
                                      batch_size=BATCH_SIZE, filename=file_val)

image_ds_test = get_predict_ds_tfrec_file(batch_size=BATCH_SIZE, filename=file_test)

image_ds_predict = get_predict_ds_tfrec_file(batch_size=BATCH_SIZE, filename=file_predict)


# ----------------------------Uncomment this and use tfrec files

# Step 5 : Build a model and compile
def jay_cnn(num_outputs):
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
        tf.keras.layers.Dense(units=num_outputs, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def mobilnetv2(num_outputs):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False,
                                                   weights="imagenet")
    mobile_net.trainable = False

    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_outputs, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


# 224
def inceptionv3_tfhub(pixels, num_classes):
    import tensorflow as tf
    import tensorflow_hub as hub
    module_selection = ("inception_v3", pixels,
                        2048)  # @param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}
    handle_base, pixels, FV_SIZE = module_selection
    MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/2".format(handle_base)
    IMAGE_SIZE = (pixels, pixels)
    print("Using {} with input size {} and output dimension {}".format(
        MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

    BATCH_SIZE = 32  # @param {type:"integer"}
    print("Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential([
        hub.KerasLayer(MODULE_HANDLE, output_shape=[FV_SIZE],
                       trainable=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + IMAGE_SIZE + (3,))
    model.summary()

    # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def mobilenet_tfhub(pixels, num_classes):
    module_selection = ("mobilenet_v2", pixels,
                        1280)  # @param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}
    handle_base, pixels, FV_SIZE = module_selection
    MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/2".format(handle_base)
    IMAGE_SIZE = (pixels, pixels)
    print("Using {} with input size {} and output dimension {}".format(
        MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

    BATCH_SIZE = 32  # @param {type:"integer"}
    print("Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential([
        hub.KerasLayer(MODULE_HANDLE, output_shape=[FV_SIZE],
                       trainable=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + IMAGE_SIZE + (3,))
    model.summary()

    # # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def tf_app_ResNet50(pixels, num_classes):
    resnet50 = tf.keras.applications.ResNet50(input_shape=(pixels, pixels, 3), include_top=False,
                                              weights="imagenet")
    resnet50.trainable = False

    model = tf.keras.Sequential([
        resnet50,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    model.build((None,) + (pixels, pixels) + (3,))
    model.summary()

    # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def tf_app_Incv3(pixels, num_classes):
    incv3 = tf.keras.applications.InceptionV3(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
    incv3.trainable = False

    model = tf.keras.Sequential([
        incv3,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    model.build((None,) + (pixels, pixels) + (3,))
    model.summary()

    # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def tf_app_Xception(pixels, num_classes):
    xception = tf.keras.applications.Xception(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
    xception.trainable = False

    model = tf.keras.Sequential([
        xception,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    model.build((None,) + (pixels, pixels) + (3,))
    model.summary()

    # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def tf_app_InceptionResNetV2(pixels, num_classes):
    incv3 = tf.keras.applications.InceptionResNetV2(input_shape=(pixels, pixels, 3), include_top=False,
                                                    weights="imagenet")
    incv3.trainable = False

    model = tf.keras.Sequential([
        incv3,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    model.build((None,) + (pixels, pixels) + (3,))
    model.summary()

    # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def tf_app_VGG19(pixels, num_classes):
    incv3 = tf.keras.applications.VGG19(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
    incv3.trainable = False

    model = tf.keras.Sequential([
        incv3,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    model.build((None,) + (pixels, pixels) + (3,))
    model.summary()

    # Training the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


# 224

def jayconvnet(batch_size, pixels, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3), input_shape=(pixels, pixels, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dense(units=600, activation="relu"))
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dense(units=500, activation="relu"))
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dense(units=400, activation="relu"))
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dense(units=300, activation="relu"))
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dense(units=200, activation="relu"))
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(1, 1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dense(units=100, activation="relu"))
    model.add(layers.Conv2D(filters=batch_size, kernel_size=(1, 1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=50, activation="relu"))
    model.add(layers.Dense(units=num_classes, activation="softmax"))

    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=[METRICS]
                  )

    return model


def train(list_func):
    for f in list_func:
        model = f(IMAGE_SIZE, NUM_CLASSES)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(TENSORBOARD_DIR, f.__name__))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_CHECKPOINT_DIR, f.__name__),
                                                        monitor='val_sparse_categorical_accuracy',
                                                        verbose=1, save_best_only=True, mode='max',
                                                        save_weights_only=True)
        history = model.fit(image_ds_train,
                            epochs=NUM_EPOCHS,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=image_ds_validate,
                            validation_steps=val_steps,
                            callbacks=[early_stop, tensorboard, checkpoint])

        print(np.mean(history.history['sparse_categorical_accuracy']))
        print(np.mean(history.history['val_sparse_categorical_accuracy']))


def predict(list_func):
    dict_results = dict()
    for f in list_func:
        model = f(IMAGE_SIZE, NUM_CLASSES)
        model.load_weights(os.path.join(MODEL_CHECKPOINT_DIR, f.__name__))
        model.compile(optimizer=OPTIMIZER,
                      loss=LOSS,
                      metrics=[METRICS]
                      )
        history_predict = model.predict(image_ds_test, steps=BATCH_SIZE)
        results = np.argmax(history_predict, axis=1)
        output = str(r2_score(y_test, results))
        dict_results[f.__name__] = output

    print(dict_results)


list_func = [inceptionv3_tfhub, mobilenet_tfhub,
             tf_app_VGG19]
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=7)

train(list_func)
predict(list_func)
