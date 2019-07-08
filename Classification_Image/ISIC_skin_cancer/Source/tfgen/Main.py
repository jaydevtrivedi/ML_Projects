#  Import all libraries
from __future__ import absolute_import, division, print_function
import tensorflow as tf

from Source.tfgen.Helper import HelperMethods

helper_methods = HelperMethods()
helper_methods.set_start_time()

from Source.tfgen.ParamsList import Params
from Source.tfgen.PathConstants import Constants

constants = Constants()
params = Params()

# Params
pixels = params.get_pixels()  # 224
IMAGE_SIZE = params.get_image_size()
BATCH_SIZE = params.get_batch_size()
metric = params.get_metric()
#

# Get data to train on
from Source.tfgen.Database import Data

data = Data()
# Get data
train_generator, valid_generator = data.get_train_valid_datagen(IMAGE_SIZE, BATCH_SIZE)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

#model_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
model_nums = [0, 1, 2]
hist_list = []
pred_list = []
for model_num in model_nums:
    model = Params.get_model(modelnum=model_num, p_model=None,
                             p_pixels=pixels, p_num_of_classes=train_generator.num_classes,
                             p_do_fine_tuning=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=metric, patience=3)
    # tensorboard = tf.keras.callbacks.TensorBoard(
    #     log_dir=constants.get_logs_dir() + constants.get_log_dir_model()[model_num])

    hist = model.fit_generator(
        train_generator,
        callbacks=[early_stop],
        epochs=1, steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps).history
    hist_list.append(hist)
    # helper_methods.saveweights(model=model,
    #                            chkptname=constants.get_log_dir_model()[model_num].replace("\\", ""))

    print("Training ends")
    predict_generator = data.get_predict_datagen(IMAGE_SIZE)
    pred = model.predict_generator(predict_generator, steps=predict_generator.samples // predict_generator.batch_size,
                                   verbose=1)
    print("Prediction ends")

    pred_list.append(pred)
    for pred in pred_list:
        helper_methods.print_predictions(pred, predict_generator.filenames)
    print("Prediction printing ends")

    import numpy as np
    print(np.mean(hist['val_accuracy']))

helper_methods.set_end_time()
print(helper_methods.get_run_time())
