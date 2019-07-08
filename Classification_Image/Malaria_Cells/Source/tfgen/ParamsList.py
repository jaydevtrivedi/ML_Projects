# Params
class Params:
    pixels = 224
    image_size = (pixels, pixels)
    batch_size = 32
    metric = 'val_accuracy'
    tf_cache_train = "cache_train.tf-data"
    tf_cache_validate = "cache_validate.tf-data"
    tfrec_train = "train.tfrec"
    tfrec_validate = "validate.tfrec"

    @staticmethod
    def get_model_0(model, pixels, num_classes, do_fine_tuning):
        return model.mobilenet_tfhub(pixels=pixels, num_classes=num_classes,
                                     do_fine_tuning=do_fine_tuning)

    @staticmethod
    def get_model_1(model, pixels, num_classes, do_fine_tuning):
        return model.inceptionv3_tfhub(pixels=pixels, num_classes=num_classes,
                                       do_fine_tuning=do_fine_tuning)

    @staticmethod
    def get_model_2(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_densenet121(pixels=pixels, num_classes=num_classes,
                                        trainable=do_fine_tuning)

    @staticmethod
    def get_model_3(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_densenet169(pixels=pixels, num_classes=num_classes,
                                        trainable=do_fine_tuning)

    @staticmethod
    def get_model_4(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_densenet201(pixels=pixels, num_classes=num_classes,
                                        trainable=do_fine_tuning)

    @staticmethod
    def get_model_5(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_inceptionresnetv2(pixels=pixels, num_classes=num_classes,
                                              trainable=do_fine_tuning)

    @staticmethod
    def get_model_6(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_inceptionv3(pixels=pixels, num_classes=num_classes,
                                        trainable=do_fine_tuning)

    @staticmethod
    def get_model_7(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_mobilenet(pixels=pixels, num_classes=num_classes,
                                      trainable=do_fine_tuning)

    @staticmethod
    def get_model_8(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_mobilenetv2(pixels=pixels, num_classes=num_classes,
                                        trainable=do_fine_tuning)

    @staticmethod
    def get_model_9(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_nasnetlarge(pixels=pixels, num_classes=num_classes,
                                        trainable=do_fine_tuning)

    @staticmethod
    def get_model_10(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_nasnetmobile(pixels=pixels, num_classes=num_classes,
                                         trainable=do_fine_tuning)

    @staticmethod
    def get_model_11(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_resnet50(pixels=pixels, num_classes=num_classes,
                                     trainable=do_fine_tuning)

    @staticmethod
    def get_model_12(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_vgg16(pixels=pixels, num_classes=num_classes,
                                  trainable=do_fine_tuning)

    @staticmethod
    def get_model_13(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_vgg19(pixels=pixels, num_classes=num_classes,
                                  trainable=do_fine_tuning)

    @staticmethod
    def get_model_14(model, pixels, num_classes, do_fine_tuning):
        return model.tf_app_xception(pixels=pixels, num_classes=num_classes,
                                     trainable=do_fine_tuning)

    @staticmethod
    def get_model_15(model, pixels, num_classes, do_fine_tuning):
        return model.jayconvnet(batch_size=self.get_batch_size(), pixels=pixels, num_classes=num_classes)

    @staticmethod
    def get_model(modelnum, p_model, p_pixels, p_num_of_classes, p_do_fine_tuning):
        from Source.tfgen.Models import Model
        model = Model
        pixels = p_pixels
        num_of_classes = p_num_of_classes
        do_fine_tuning = p_do_fine_tuning

        if modelnum == 0:
            return Params.get_model_0(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 1:
            return Params.get_model_1(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 2:
            return Params.get_model_2(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 3:
            return Params.get_model_3(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 4:
            return Params.get_model_4(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 5:
            return Params.get_model_5(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 6:
            return Params.get_model_6(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 7:
            return Params.get_model_7(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 8:
            return Params.get_model_8(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 9:
            return Params.get_model_9(model=model, pixels=pixels, num_classes=num_of_classes,
                                      do_fine_tuning=do_fine_tuning)
        elif modelnum == 10:
            return Params.get_model_10(model=model, pixels=pixels, num_classes=num_of_classes,
                                       do_fine_tuning=do_fine_tuning)
        elif modelnum == 11:
            return Params.get_model_11(model=model, pixels=pixels, num_classes=num_of_classes,
                                       do_fine_tuning=do_fine_tuning)
        elif modelnum == 12:
            return Params.get_model_12(model=model, pixels=pixels, num_classes=num_of_classes,
                                       do_fine_tuning=do_fine_tuning)
        elif modelnum == 13:
            return Params.get_model_13(model=model, pixels=pixels, num_classes=num_of_classes,
                                       do_fine_tuning=do_fine_tuning)
        elif modelnum == 14:
            return Params.get_model_14(model=model, pixels=pixels, num_classes=num_of_classes,
                                       do_fine_tuning=do_fine_tuning)
        elif modelnum == 15:
            return Params.get_model_15(model=model, pixels=pixels, num_classes=num_of_classes,
                                       do_fine_tuning=do_fine_tuning)

    def get_pixels(self):
        return self.pixels

    def get_image_size(self):
        return self.image_size

    def get_batch_size(self):
        return self.batch_size

    def get_metric(self):
        return self.metric

    def get_tf_cache_train_filename(self):
        return self.tf_cache_train

    def get_tf_cache_validate_filename(self):
        return self.tf_cache_validate

    def get_tfrec_train_filename(self):
        return self.tfrec_train

    def get_tfrec_validate_filename(self):
        return self.tfrec_validate
