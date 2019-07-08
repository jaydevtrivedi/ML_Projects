class Constants:
    #  Always reference from Sources root
    database_dir = "./datasets"
    predict_dir = database_dir + "/predict_files"
    logs_dir = "C:\\Users\\Jaydev\\Documents\\GitHub\\repos\\deep_learning\\ISIC_skin_cancer\\logs"
    log_dir_model = ["\\tf_hub\\mobilenet", "\\tf_hub\\inceptionv3", "\\keras_apps\\densenet121",
                     "\\keras_apps\\densenet169",
                     "\\keras_apps\\densenet201", "\\keras_apps\\inceptionresnetv2",
                     "\\keras_apps\\inceptionv3", "\\keras_apps\\mobilenet", "\\keras_apps\\mobilenetv2",
                     "\\keras_apps\\nasnetlarge",
                     "\\keras_apps\\nasnetmobile", "\\keras_apps\\resnet50", "\\keras_apps\\vgg16",
                     "\\keras_apps\\vgg19", "\\keras_apps\\Xception"]

    tfdata_dir = "./tfdata_dir"
    tfrec_dir = tfdata_dir + "/tfrec_dir"
    tfcache_dir = tfdata_dir + "/tfcache_dir"

    # ISIC skin cancer
    # win_path_all_imgs = "C:\\Users\\Jaydev\\Documents\\GitHub\\repos\\deep_learning\\ISIC_skin_cancer\\datasets\\all_imgs"
    # win_path_pred_imgs = "C:\\Users\\Jaydev\\Documents\\GitHub\\repos\\deep_learning\\ISIC_skin_cancer\\datasets\\predict_files"

    # Malaria test
    win_path_all_imgs = "C:\\Users\\Jaydev\\Documents\\GitHub\\repos\\deep_learning\\ISIC_skin_cancer\\datasets\\all_imgs_malaria"
    win_path_pred_imgs = "C:\\Users\\Jaydev\\Documents\\GitHub\\repos\\deep_learning\\ISIC_skin_cancer\\datasets\\predict_files_malaria"

    # ------------------------------------------------------------------------------

    def get_database_dir_path(self):
        return self.database_dir

    def get_predict_dir_path(self):
        return self.predict_dir

    def get_tfdata_dir_path(self):
        return self.tfdata_dir

    def get_tfrec_dir_path(self):
        return self.tfrec_dir

    def get_tfcache_dir_path(self):
        return self.tfcache_dir

    def get_all_imgs_dir_path_win(self):
        return self.win_path_all_imgs

    def get_win_path_pred_imgs(self):
        return self.win_path_pred_imgs

    def get_logs_dir(self):
        return self.logs_dir

    def get_log_dir_model(self):
        return self.log_dir_model
