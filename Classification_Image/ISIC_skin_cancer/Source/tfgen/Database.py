class Data:
    #  Always reference from Sources root

# ------------------------------------------------------------------------------

    def get_train_valid_datagen(self, IMAGE_SIZE, BATCH_SIZE):
        import tensorflow as tf
        from Source.tfgen.PathConstants import Constants
        constants = Constants()
        win_path_all_imgs = constants.get_all_imgs_dir_path_win()

        data_dir = tf.keras.utils.get_file(
            win_path_all_imgs,
            origin=None,
            untar=False)

        datagen_kwargs = dict(rescale=1. / 255, validation_split=.20)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagen_kwargs)
        valid_generator = valid_datagen.flow_from_directory(
            data_dir, subset="validation", shuffle=False,
            target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=42)

        do_data_augmentation = False  # @param {type:"boolean"}
        if do_data_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2, height_shift_range=0.2,
                shear_range=0.2, zoom_range=0.2,
                **datagen_kwargs)
        else:
            train_datagen = valid_datagen
        train_generator = train_datagen.flow_from_directory(
            data_dir, subset="training", shuffle=True,
            target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=42)

        return train_generator, valid_generator

    def get_predict_datagen(self, IMAGE_SIZE):
        import tensorflow as tf
        from Source.tfgen.PathConstants import Constants
        constants = Constants()
        win_path_pred_imgs = constants.get_win_path_pred_imgs()

        predict_dir = tf.keras.utils.get_file(
            win_path_pred_imgs,
            origin=None,
            untar=False)
        predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        predict_generator = predict_datagen.flow_from_directory(
            predict_dir, class_mode=None, shuffle=True,
            target_size=IMAGE_SIZE, batch_size=1)

        return predict_generator

# ------------------------------------------------------------------------------

