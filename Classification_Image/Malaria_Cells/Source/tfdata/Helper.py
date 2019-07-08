class HelperMethods:
    saved_models_dir = './savedmodels'

    def starttime(self):
        import datetime
        self.starttime = datetime.datetime.now()

    def endtime(self):
        import datetime
        self.endtime = datetime.datetime.now()

    def getruntime(self):
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
    def load_and_preprocess_image(self, *args):
        import tensorflow as tf
        from Source.tfdata.ParamList import Params
        params = Params()

        multiple_args = len(args) > 1
        if multiple_args:
            path = args[0]
            label = args[1]
        else:
            path = args[0]

        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [params.get_pixels(), params.get_pixels()])
        image /= 255.0

        if multiple_args:
            return image, label
        else:
            return image

    def load_and_preprocess_image_mobilnet(self, path, label):
        import tensorflow as tf
        from Source.tfdata.ParamList import Params
        params = Params()

        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [params.get_pixels(), params.get_pixels()])
        image /= 255.0
        image = 2 * image - 1

        return image, label

    def parse(self, x):
        import tensorflow as tf
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        # result = tf.reshape(result, [192, 192, 3])
        return result

    def savemodel(self, model, modelname):
        model.save(self.saved_models_dir + "/" + modelname)

    def saveweights(self, model, chkptname):
        model.save_weights(self.saved_models_dir + "/" + chkptname)

    def print_predictions(self, label_map_dictionary, predictions, prediction_file_list):
        import numpy as np
        index_to_label = {}
        for label in label_map_dictionary.keys():
            index_to_label[label_map_dictionary.get(label)] = label

        for counter in range(len(predictions) - 1):
            print("Filename -  {} is {}",
                  (prediction_file_list[counter], index_to_label[np.argmax(predictions[counter])]))
