class Model:

    @staticmethod
    def mobilenet_tfhub(pixels, num_classes, do_fine_tuning):
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
        model.build((None,) + (pixels, pixels) + (3,))
        model.summary()

        # Training the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def inceptionv3_tfhub(pixels, num_classes, do_fine_tuning):
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
        model.build((None,) + (pixels, pixels) + (3,))
        model.summary()

        # Training the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_densenet121(pixels, num_classes, trainable):
        import tensorflow as tf
        densenet121 = tf.keras.applications.DenseNet121(input_shape=(pixels, pixels, 3), include_top=False,
                                                        weights="imagenet")
        densenet121.trainable = trainable

        model = tf.keras.Sequential([
            densenet121,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_densenet169(pixels, num_classes, trainable):
        import tensorflow as tf
        densenet169 = tf.keras.applications.DenseNet169(input_shape=(pixels, pixels, 3), include_top=False,
                                                        weights="imagenet")
        densenet169.trainable = trainable

        model = tf.keras.Sequential([
            densenet169,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_densenet201(pixels, num_classes, trainable):
        import tensorflow as tf
        densenet201 = tf.keras.applications.DenseNet201(input_shape=(pixels, pixels, 3), include_top=False,
                                                        weights="imagenet")
        densenet201.trainable = trainable

        model = tf.keras.Sequential([
            densenet201,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_inceptionresnetv2(pixels, num_classes, trainable):
        import tensorflow as tf
        inceptionresnetv2 = tf.keras.applications.InceptionResNetV2(input_shape=(pixels, pixels, 3), include_top=False,
                                                                    weights="imagenet")
        inceptionresnetv2.trainable = trainable

        model = tf.keras.Sequential([
            inceptionresnetv2,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_inceptionv3(pixels, num_classes, trainable):
        import tensorflow as tf
        inceptionv3 = tf.keras.applications.InceptionV3(input_shape=(pixels, pixels, 3), include_top=False,
                                                        weights="imagenet")
        inceptionv3.trainable = trainable

        model = tf.keras.Sequential([
            inceptionv3,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_mobilenet(pixels, num_classes, trainable):
        import tensorflow as tf
        mobilenet = tf.keras.applications.MobileNet(input_shape=(pixels, pixels, 3), include_top=False,
                                                    weights="imagenet")
        mobilenet.trainable = trainable

        model = tf.keras.Sequential([
            mobilenet,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_mobilenetv2(pixels, num_classes, trainable):
        import tensorflow as tf
        mobilenetv2 = tf.keras.applications.MobileNetV2(input_shape=(pixels, pixels, 3), include_top=False,
                                                        weights="imagenet")
        mobilenetv2.trainable = trainable

        model = tf.keras.Sequential([
            mobilenetv2,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_nasnetlarge(pixels, num_classes, trainable):
        import tensorflow as tf
        nasnetlarge = tf.keras.applications.NASNetLarge(input_shape=(pixels, pixels, 3), include_top=False,
                                                        weights="imagenet")
        nasnetlarge.trainable = trainable

        model = tf.keras.Sequential([
            nasnetlarge,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_nasnetmobile(pixels, num_classes, trainable):
        import tensorflow as tf
        nasnetmobile = tf.keras.applications.NASNetMobile(input_shape=(pixels, pixels, 3), include_top=False,
                                                          weights="imagenet")
        nasnetmobile.trainable = trainable

        model = tf.keras.Sequential([
            nasnetmobile,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_resnet50(pixels, num_classes, trainable):
        import tensorflow as tf
        resnet50 = tf.keras.applications.ResNet50(input_shape=(pixels, pixels, 3), include_top=False,
                                                  weights="imagenet")
        resnet50.trainable = trainable

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
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_vgg16(pixels, num_classes, trainable):
        import tensorflow as tf
        vgg16 = tf.keras.applications.VGG16(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
        vgg16.trainable = trainable

        model = tf.keras.Sequential([
            vgg16,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_vgg19(pixels, num_classes, trainable):
        import tensorflow as tf
        vgg19 = tf.keras.applications.VGG19(input_shape=(pixels, pixels, 3), include_top=False, weights="imagenet")
        vgg19.trainable = trainable

        model = tf.keras.Sequential([
            vgg19,
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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def tf_app_xception(pixels, num_classes, trainable):
        import tensorflow as tf
        xception = tf.keras.applications.Xception(input_shape=(pixels, pixels, 3), include_top=False,
                                                  weights="imagenet")
        xception.trainable = trainable

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
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        return model

    @staticmethod
    def jayconvnet(batch_size, pixels, num_classes):
        import tensorflow as tf
        from tensorflow.keras import models
        from tensorflow.keras import layers

        model = models.Sequential()
        model.add(
            layers.Conv2D(filters=32, kernel_size=3, strides=1, activation="relu", input_shape=(pixels,pixels)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(
            layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(
            layers.Conv2D(filters=128, kernel_size=3, strides=1, activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=128, activation="relu"))
        model.add(layers.Dense(units=num_classes, activation="softmax"))

        # Training the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.0045, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        # model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
        #               loss=tf.keras.losses.CategoricalCrossentropy(),
        #               metrics=[tf.keras.metrics.Accuracy()])

        return model

