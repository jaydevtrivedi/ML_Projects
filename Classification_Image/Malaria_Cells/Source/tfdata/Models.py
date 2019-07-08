class Model:

    def mobilnetv2(self, num_outputs):
        import tensorflow as tf
        mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False, weights="imagenet")
        mobile_net.trainable = False

        model = tf.keras.Sequential([
            mobile_net,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_outputs, activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    def mobilenet_tfhub(self, pixels, num_classes, do_fine_tuning):
        import tensorflow as tf
        import tensorflow_hub as hub
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
                           trainable=do_fine_tuning),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])
        model.build((None,) + IMAGE_SIZE + (3,))
        model.summary()

        # # Training the model
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

        return model

    def inceptionv3_tfhub(self, pixels, num_classes, do_fine_tuning):
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
                           trainable=do_fine_tuning),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])
        model.build((None,) + IMAGE_SIZE + (3,))
        model.summary()

        # Training the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

        return model

    def tf_app_ResNet50(self, pixels, num_classes):
        import tensorflow as tf
        resnet50 = tf.keras.applications.ResNet50(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
        resnet50.trainable = False

        model = tf.keras.Sequential([
            resnet50,
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])

        model.build((None,) + (pixels,pixels) + (3,))
        model.summary()

        # Training the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

        return model

    def tf_app_Incv3(self, pixels, num_classes):
        import tensorflow as tf
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
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

        return model

    def tf_app_Xception(self, pixels, num_classes):
        import tensorflow as tf
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
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

        return model

    # np.mean(hist['val_accuracy'])
    # 0.7409611
    def tf_app_InceptionResNetV2(self, pixels, num_classes):
        import tensorflow as tf
        incv3 = tf.keras.applications.InceptionResNetV2(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
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
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

        return model

    # np.mean(hist['val_accuracy'])
    # 0.68867606
    def tf_app_VGG19(self, pixels, num_classes):
        import tensorflow as tf
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
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])

        return model

    def jayconvnet(self, batch_size, pixels, num_classes):
        import tensorflow as tf
        from tensorflow.keras import models
        from tensorflow.keras import layers

        model = models.Sequential()
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3), input_shape=(pixels, pixels, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dense(units=600, activation="relu"))
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3),  activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dense(units=500, activation="relu"))
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3),  activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dense(units=400, activation="relu"))
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3),  activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dense(units=300, activation="relu"))
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(3, 3),  activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dense(units=200, activation="relu"))
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(1, 1),  activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dense(units=100, activation="relu"))
        model.add(layers.Conv2D(filters=batch_size, kernel_size=(1, 1),  activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=50, activation="relu"))
        model.add(layers.Dense(units=num_classes, activation="softmax"))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

        return model
