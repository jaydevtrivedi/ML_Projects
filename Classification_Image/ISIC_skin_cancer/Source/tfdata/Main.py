from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from Source.tfdata.Helper import HelperMethods
from Source.tfdata.Database import Data
from Source.tfdata.ParamList import Params

helpermethods = HelperMethods()
helpermethods.starttime()
data = Data()
params_list = Params()

# Get training, test and validation batches
test_size = 0.2
random_state = 0
from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(data.get_all_imgs_paths_no_class(),
                                                            data.get_all_imgs_labels(),
                                                            test_size=test_size, random_state=random_state)

# image_ds_train = data.get_ds_tensor_slices(X_train=X_train, y_train=y_train, batch_size=params_list.get_batch_size(),
#                                            load_preprocess_image=helpermethods.load_and_preprocess_image)
#
# image_ds_validate = data.get_ds_tensor_slices(X_train=X_train, y_train=y_train,
#                                               batch_size=params_list.get_batch_size(),
#                                               load_preprocess_image=helpermethods.load_and_preprocess_image)

# image_ds_train = data.get_ds_tfdata_cache(X_train=X_train, y_train=y_train, batch_size=params_list.get_batch_size(),
#                                           load_preprocess_image=helpermethods.load_and_preprocess_image,
#                                           filename=params_list.get_tf_cache_train_filename())
#
# image_ds_validate = data.get_ds_tfdata_cache(X_train=X_train, y_train=y_train,
#                                              batch_size=params_list.get_batch_size(),
#                                              load_preprocess_image=helpermethods.load_and_preprocess_image,
#                                              filename=params_list.get_tf_cache_test_filename())

tfrec_train_file_ = data.write_ds_tfrec_file(X_train=X_train, y_train=y_train,
                                             load_preprocess_image=helpermethods.load_and_preprocess_image,
                                             filename=params_list.get_tfrec_train_filename())

tfrec_validate_file = data.write_ds_tfrec_file(X_train=X_train, y_train=y_train,
                                               load_preprocess_image=helpermethods.load_and_preprocess_image,
                                               filename=params_list.get_tfrec_validate_filename())

image_ds_train = data.get_ds_tfrec_file(y_train=y_train, batch_size=params_list.get_batch_size(),
                                        load_preprocess_image=helpermethods.load_and_preprocess_image,
                                        filename=params_list.get_tfrec_train_filename())
image_ds_validate = data.get_ds_tfrec_file(y_train=y_train, batch_size=params_list.get_batch_size(),
                                           load_preprocess_image=helpermethods.load_and_preprocess_image,
                                           filename=params_list.get_tfrec_validate_filename())

# Creating a dataset from labels
steps_per_epoch = len(X_train) // params_list.get_batch_size()
validation_steps = len(X_validate) // params_list.get_batch_size()

# CREATE THE MODEL MOBILENET
early_stop_patience = 5
training_epochs = 1
from Source.tfdata.Models import Model

model = Model().jayconvnet(batch_size=params_list.get_batch_size(), pixels=params_list.get_pixels(),
                           num_classes=data.num_outputs)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=early_stop_patience)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
history = model.fit(x=image_ds_train, validation_data=image_ds_validate, epochs=training_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps, callbacks=[early_stop, tensorboard])
#
# X_predict = data.get_all_imgs_predict()
# image_ds_predict = data.get_ds_predict_images(X_predict, 1,
#                                               load_preprocess_image=helpermethods.load_and_preprocess_image)
# pred = model.predict(image_ds_predict, steps=(len(X_predict)), verbose=1)
# helpermethods.print_predictions(data.dict, pred, X_predict)
helpermethods.endtime()

print(helpermethods.getruntime())
