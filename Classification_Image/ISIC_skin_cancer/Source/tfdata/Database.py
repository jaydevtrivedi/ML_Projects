class Data:
    def __init__(self):
        self.image_count = 0
        self.num_outputs = 0
        self.dict = {}

# Global Variables
    def get_imagecount(self):
        return self.image_count

    def get_num_outputs(self):
        return self.num_outputs

    def get_label_to_index_dict(self):
        return self.dict
# Global Variables

    def get_hmnist_metadata_dataframe(self):
        import pandas as pd
        from Source.tfdata.PathConstants import Constants
        hmnist_metadata = pd.read_csv(Constants().get_hmnist_metadata_file_path())
        return hmnist_metadata

# ------------------------------------------------------------------------------

    def get_all_imgs_paths_no_class(self):
        import os
        from Source.tfdata.PathConstants import Constants
        all_imgs_no_class = Constants().get_all_imgs_no_class_dir_path()

        all_images = os.listdir(all_imgs_no_class)
        # Get path to all images
        all_image_paths = []
        for image_names in all_images:
            all_image_paths.append(all_imgs_no_class + "/" + image_names)
        self.image_count = len(all_image_paths)
        return all_image_paths

    def get_all_imgs_labels(self):
        import numpy as np
        from Source.tfdata.PathConstants import Constants
        hmnist_metadata = self.get_hmnist_metadata_dataframe()

        # all_image_labels = y
        hmnist_metadata = hmnist_metadata.sort_values(by=['image_id'])
        all_image_labels = hmnist_metadata['dx']

        label_names = np.sort(all_image_labels.unique())
        self.num_outputs = len(label_names)

        # Creating a label to index dictionary
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        self.dict = label_to_index

        # Converting all image labels to integers
        all_image_labels = [label_to_index[path]
                            for path in all_image_labels]
        return all_image_labels

    def get_all_imgs_predict(self):
        # Get all files to predict
        import os
        from Source.tfdata.PathConstants import Constants
        predict_image_dir_path = Constants().get_predict_dir_path()
        predict_image_paths = os.listdir(predict_image_dir_path)

        # Get path to all images
        X_predict = []
        for image_names in predict_image_paths:
            X_predict.append(predict_image_dir_path + "/" + image_names)
        return X_predict

# ------------------------------------------------------------------------------

    def get_ds_tensor_slices(self, X_train, y_train, batch_size, load_preprocess_image):
        import tensorflow as tf
        ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        ds = ds.map(load_preprocess_image)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=batch_size)
        return ds

    def get_ds_predict_images(self, X_predict, batch_size, load_preprocess_image):
        import tensorflow as tf
        ds = tf.data.Dataset.from_tensor_slices(X_predict)
        ds = ds.map(load_preprocess_image)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=batch_size)
        return ds

    # tensorflow bug #18266 lock file for validationcache file
    def get_ds_tfdata_cache(self, X_train, y_train, batch_size, load_preprocess_image, filename):
        import tensorflow as tf
        from Source.tfdata.PathConstants import Constants
        cachedir = Constants().get_tfcache_dir_path()
        ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        ds = ds.map(load_preprocess_image)
        ds = ds.cache(filename=cachedir + "/" + filename)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=batch_size)
        return ds

    def write_ds_tfrec_file(self, X_train, y_train, load_preprocess_image, filename):
        import tensorflow as tf
        from Source.tfdata.PathConstants import Constants
        tfrec_file_path = Constants().get_tfrec_dir_path() + "/" + filename

        image_ds = tf.data.Dataset.from_tensor_slices(X_train).map(tf.io.read_file)
        tfrec_file = tf.data.experimental.TFRecordWriter(filename=tfrec_file_path)
        tfrec_file.write(image_ds)
        return tfrec_file

    def get_ds_tfrec_file(self, y_train, batch_size, load_preprocess_image, filename):
        import tensorflow as tf
        from Source.tfdata.PathConstants import Constants
        tfrec_file_path = Constants().get_tfrec_dir_path() + "/" + filename

        image_ds = tf.data.TFRecordDataset(tfrec_file_path).map(load_preprocess_image)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.int64))
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(buffer_size=batch_size)
        return ds
