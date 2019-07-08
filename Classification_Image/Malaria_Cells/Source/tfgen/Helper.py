import tensorflow as tf


class HelperMethods:
    saved_models_dir = './savedmodels'
    saved_weights_dir = './savedweights'

    def __init__(self):
        self.starttime = 0
        self.endtime = 0

    def set_start_time(self):
        import datetime
        self.starttime = datetime.datetime.now()

    def set_end_time(self):
        import datetime
        self.endtime = datetime.datetime.now()

    def get_run_time(self):
        print("Runtime was {}", self.endtime - self.starttime)

    def timeit(self, ds, steps_per_epoch, BATCH_SIZE):
        import time
        batches = 2 * steps_per_epoch + 1
        overall_start = time.time()
        # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
        # before starting the timer
        it = iter(ds.take(batches + 1))
        next(it)

        start = time.time()
        for i, (images, labels) in enumerate(it):
            if i % 10 == 0:
                print('.', end='')
        print()
        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(batches, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE * batches / duration))
        print("Total time: {}s".format(end - overall_start))

    # Make the dataset
    def preprocess_image(self, image):
        import tensorflow as tf
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image /= 255.0

        return image

    def load_and_preprocess_image(self, path, label):
        import tensorflow as tf
        image = tf.io.read_file(path)
        return self.preprocess_image(image), label

    def preprocess_image_mobilnet(self, image):
        import tensorflow as tf
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image /= 255.0
        image = 2 * image - 1
        return image

    def load_and_preprocess_image_mobilnet(self, path, label):
        import tensorflow as tf
        image = tf.io.read_file(path)
        return self.preprocess_image_mobilnet(image), label

    def parse(self, x):
        import tensorflow as tf
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        # result = tf.reshape(result, [192, 192, 3])
        return result

    def savemodel(self, model, modelname):
        model.save(self.saved_models_dir + "/" + modelname)

    def saveweights(self, model, chkptname):
        model.save_weights(self.saved_weights_dir + "/" + chkptname)

    def print_predictions(self, predictions, filenames):
        import os
        import numpy as np
        from Source.tfgen.PathConstants import Constants
        constants = Constants()
        classes_path = os.listdir(constants.get_all_imgs_dir_path_win())
        index_to_label = {}
        for label in range(len(classes_path)):
            index_to_label[label] = classes_path[label]

        for counter in range(len(predictions)):
            print("Filename -  {} is {}",
                  (filenames[counter].split("\\")[1], index_to_label[np.argmax(predictions[counter])]))
